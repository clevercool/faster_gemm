

__global__ void ori_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, int iteration) {
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;
    
    float alpha = 2.0f;
    float beta = 2.0f;

    for (int loop = 0; loop < iteration; loop++) {
        // Partial results
        float c[TILE_N];
        for (int i = 0; i < TILE_N; i++)
            c[i] = 0.0f;
        int mid = threadIdx.y * blockDim.x + threadIdx.x; //flattened id
        int m = blockIdx.x * TILE_M + mid;
        int n = blockIdx.y * TILE_N + threadIdx.x;
        __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

        for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
        {
            float a;
            b_s[threadIdx.y][threadIdx.x] = B[n + (i + threadIdx.y) * ldb];
            __syncthreads();
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            __syncthreads();
        }
        int t = ldc * blockIdx.y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}

__global__ void ori_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, int iteration, unsigned int*smid) {
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;
    
    float alpha = 2.0f;
    float beta = 2.0f;

	if(threadIdx.x == 0)
	{
        smid[blockIdx.y * gridDim.x + blockIdx.x] = 0;
		unsigned int ret = 0;
		asm("mov.u32 %0, %smid;" : "=r"(ret));
		smid[blockIdx.y * gridDim.x + blockIdx.x] = ret;
	}

    for (int loop = 0; loop < iteration; loop++) {
        // Partial results
        float c[TILE_N];
        for (int i = 0; i < TILE_N; i++)
            c[i] = 0.0f;
        int mid = threadIdx.y * blockDim.x + threadIdx.x; //flattened id
        int m = blockIdx.x * TILE_M + mid;
        int n = blockIdx.y * TILE_N + threadIdx.x;
        __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

        for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
        {
            float a;
            b_s[threadIdx.y][threadIdx.x] = B[n + (i + threadIdx.y) * ldb];
            __syncthreads();
            for (int j = 0; j < TILE_TB_HEIGHT; j++)
            {
                a = A[m + (i + j) * lda];
                for (int kk = 0; kk < TILE_N; kk++)
                    c[kk] += a * b_s[j][kk];
            }
            __syncthreads();
        }
        int t = ldc * blockIdx.y * TILE_N + m;
        for (int i = 0; i < TILE_N; i++)
        {
            C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
        }
    }
}


__global__ void pers_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K,
                        int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, int iteration, unsigned int*smid) {
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;

    float alpha = 2.0f;
    float beta = 2.0f;

	if(threadIdx.x == 0)
	{
        smid[blockIdx.y * gridDim.x + blockIdx.x] = 0;
		unsigned int ret = 0;
		asm("mov.u32 %0, %smid;" : "=r"(ret));
		smid[blockIdx.y * gridDim.x + blockIdx.x] = ret;
    }
    
    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x % block_dimension_x;
    int thread_id_y = threadIdx.x / block_dimension_x;

    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y)
        {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
            // Partial results
            float c[TILE_N];
            for (int i = 0; i < TILE_N; i++)
                c[i] = 0.0f;
            int mid = threadIdx.x;
            int m = block_id_x * TILE_M + mid;
            int n = block_id_y * TILE_N + thread_id_x;
            

            for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
            {
                float a;
                b_s[thread_id_y][thread_id_x] = B[n + (i + thread_id_y) * ldb];
                __syncthreads();
                for (int j = 0; j < TILE_TB_HEIGHT; j++)
                {
                    a = A[m + (i + j) * lda];
                    for (int kk = 0; kk < TILE_N; kk++)
                        c[kk] += a * b_s[j][kk];
                }
                __syncthreads();
            }
            int t = ldc * block_id_y * TILE_N + m;
            for (int i = 0; i < TILE_N; i++)
            {
                C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
            }
        }
    }
}


__global__ void pers_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K,
                        int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, int iteration) {
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;

    float alpha = 2.0f;
    float beta = 2.0f;

    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x % block_dimension_x;
    int thread_id_y = threadIdx.x / block_dimension_x;

    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y)
        {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
            // Partial results
            float c[TILE_N];
            for (int i = 0; i < TILE_N; i++)
                c[i] = 0.0f;
            int mid = threadIdx.x;
            int m = block_id_x * TILE_M + mid;
            int n = block_id_y * TILE_N + thread_id_x;
            

            for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
            {
                float a;
                b_s[thread_id_y][thread_id_x] = B[n + (i + thread_id_y) * ldb];
                __syncthreads();
                for (int j = 0; j < TILE_TB_HEIGHT; j++)
                {
                    a = A[m + (i + j) * lda];
                    for (int kk = 0; kk < TILE_N; kk++)
                        c[kk] += a * b_s[j][kk];
                }
                __syncthreads();
            }
            int t = ldc * block_id_y * TILE_N + m;
            for (int i = 0; i < TILE_N; i++)
            {
                C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
            }
        }
    }
}


__device__ void mix_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
                    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y,
                    int iteration, int thread_step) {
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;

    float alpha = 2.0f;
    float beta = 2.0f;

    unsigned int block_pos = blockIdx.x;
    int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

    for (;; block_pos += SGEMM_GRID_DIM) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
            // Partial results
            float c[TILE_N];
            for (int i = 0; i < TILE_N; i++)
                c[i] = 0.0f;
            int mid = (threadIdx.x - thread_step);
            int m = block_id_x * TILE_M + mid;
            int n = block_id_y * TILE_N + thread_id_x;
            

            for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
            {
                float a;
                b_s[thread_id_y][thread_id_x] = B[n + (i + thread_id_y) * ldb];
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(0), "r"(128) : "memory");
                for (int j = 0; j < TILE_TB_HEIGHT; j++)
                {
                    a = A[m + (i + j) * lda];
                    for (int kk = 0; kk < TILE_N; kk++)
                        c[kk] += a * b_s[j][kk];
                }
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(0), "r"(128) : "memory");
            }
            int t = ldc * block_id_y * TILE_N + m;
            for (int i = 0; i < TILE_N; i++)
            {
                C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
            }
        }
    }
}


__device__ void mixx_sgemm(float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
                    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y,
                    int iteration, int thread_step) {
    int lda = NORMAL_M;
    int ldb = NORMAL_N;
    int ldc = NORMAL_M;

    float alpha = 2.0f;
    float beta = 2.0f;

    unsigned int block_pos = blockIdx.x + 68 * (thread_step / (block_dimension_x * block_dimension_y));
    int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    int tmp = thread_step / (block_dimension_x * block_dimension_y);

    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];

    for (;; block_pos += 68 * 2) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
            // Partial results
            float c[TILE_N];
            for (int i = 0; i < TILE_N; i++)
                c[i] = 0.0f;
            int mid = (threadIdx.x - thread_step);
            int m = block_id_x * TILE_M + mid;
            int n = block_id_y * TILE_N + thread_id_x;
            

            for (int i = 0; i < NORMAL_K; i += TILE_TB_HEIGHT)
            {
                float a;
                b_s[thread_id_y][thread_id_x] = B[n + (i + thread_id_y) * ldb];
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(tmp), "r"(128) : "memory");
                for (int j = 0; j < TILE_TB_HEIGHT; j++)
                {
                    a = A[m + (i + j) * lda];
                    for (int kk = 0; kk < TILE_N; kk++)
                        c[kk] += a * b_s[j][kk];
                }
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(tmp), "r"(128) : "memory");
            }
            int t = ldc * block_id_y * TILE_N + m;
            for (int i = 0; i < TILE_N; i++)
            {
                C[t + i * ldc] = C[t + i * ldc] * beta + alpha * c[i];
            }
        }
    }
}


__global__ void new_sgemm(
    float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K,
    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, int iteration) {
    if (threadIdx.x < 128) {
        // mix_sgemm(A, B, C, NORMAL_M, NORMAL_N, NORMAL_K, grid_dimension_x, grid_dimension_y, block_dimension_x, block_dimension_y, iteration);
        mixx_sgemm(A, B, C, NORMAL_M, NORMAL_N, NORMAL_K, grid_dimension_x, grid_dimension_y, block_dimension_x, block_dimension_y, iteration, 0);
    } else {
        mixx_sgemm(A, B, C, NORMAL_M, NORMAL_N, NORMAL_K, grid_dimension_x, grid_dimension_y, block_dimension_x, block_dimension_y, 
            iteration, 128);
    }
}
