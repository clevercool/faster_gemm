#include <assert.h>
#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <mma.h>
using namespace nvcuda;

#include "tzgemm_mix.cu"

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}

#define checkKernelErrors(expr)                                   \
    do                                                            \
    {                                                             \
        expr;                                                     \
                                                                  \
        cudaError_t __err = cudaGetLastError();                   \
        if (__err != cudaSuccess)                                 \
        {                                                         \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
                   cudaGetErrorString(__err));                    \
            abort();                                              \
        }                                                         \
    } while (0)



int main(int argc, char* argv[]) {
    int pers_sgemm_block = 2;
    int sgemm_iter = 66;
    int wmma_iter = 3600;
    int M_INPUT = 16 * 32 * 8;
	int N_INPUT = 16 * 4 * 48;
	int K_INPUT = 16 * 4 * 12;
    if (argc == 7) {
        pers_sgemm_block = atoi(argv[1]);
        sgemm_iter = atoi(argv[2]);
        wmma_iter = atoi(argv[3]);
        M_INPUT = atoi(argv[4]);
        N_INPUT = atoi(argv[5]);
        K_INPUT = atoi(argv[6]);
    } 

    int M_GLOBAL = (M_INPUT < 64) ? 64 : (M_INPUT / 64) * 64;
	int N_GLOBAL = (N_INPUT < 64) ? 64 : (N_INPUT / 64) * 64;
	int K_GLOBAL = (K_INPUT < 64) ? 64 : (K_INPUT / 64) * 64;

	int M_TILES = M_GLOBAL / WMMA_M;
	int N_TILES = N_GLOBAL / WMMA_N;
	int K_TILES = K_GLOBAL / WMMA_K;

    float kernel_time;
    curandGenerator_t gen;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));

    // wmma variables
    // ----------------------------------------------------------------------------------------------------------------------
    float *ori_host_A = NULL;
	float *ori_host_B = NULL;
	float *ori_result_C = NULL;
	float *mix_result_C = NULL;

	half *ori_wmma_A = NULL;
	half *ori_wmma_B = NULL;
	float *ori_wmma_C = NULL;

    half *mix_wmma_A = NULL;
	half *mix_wmma_B = NULL;
	float *mix_wmma_C = NULL;

	// ori_host_A = (half *)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
	// ori_host_B = (half *)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
	ori_result_C = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
	mix_result_C = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

	// init_host_matrices(ori_host_A, ori_host_B);
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_A), sizeof(half) * M_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_B), sizeof(half) * N_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_C), sizeof(float) * M_GLOBAL * N_GLOBAL));
    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&mix_wmma_A), sizeof(half) * M_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&mix_wmma_B), sizeof(half) * N_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&mix_wmma_C), sizeof(float) * M_GLOBAL * N_GLOBAL));

	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_A), sizeof(float) * M_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_B), sizeof(float) * N_GLOBAL * K_GLOBAL));

    curandErrCheck(curandGenerateUniform(gen, ori_host_A, M_GLOBAL * K_GLOBAL));
    curandErrCheck(curandGenerateUniform(gen, ori_host_B, N_GLOBAL * K_GLOBAL));
    // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
    convertFp32ToFp16 <<< (M_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_A, ori_host_A, M_GLOBAL * K_GLOBAL);
    convertFp32ToFp16 <<< (N_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_B, ori_host_B, N_GLOBAL * K_GLOBAL);
	cudaErrCheck(cudaMemcpy(mix_wmma_A, ori_wmma_A, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyDeviceToDevice));
	cudaErrCheck(cudaMemcpy(mix_wmma_B, ori_wmma_B, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyDeviceToDevice));

	assert(((unsigned long long)ori_wmma_A) % 128 == 0);
	assert(((unsigned long long)ori_wmma_B) % 128 == 0);
	assert(((unsigned long long)ori_wmma_C) % 128 == 0);
    assert(((unsigned long long)mix_wmma_A) % 128 == 0);
	assert(((unsigned long long)mix_wmma_B) % 128 == 0);
	assert(((unsigned long long)mix_wmma_C) % 128 == 0);

	// cudaErrCheck(cudaMemcpy(ori_wmma_A, ori_host_A, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
	// cudaErrCheck(cudaMemcpy(mix_wmma_A, ori_host_A, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
	// cudaErrCheck(cudaMemcpy(ori_wmma_B, ori_host_B, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
	// cudaErrCheck(cudaMemcpy(mix_wmma_B, ori_host_B, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
	cudaErrCheck(cudaMemset(ori_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));
	cudaErrCheck(cudaMemset(mix_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // start solo running 
    // ----------------------------------------------------------------------------------------------------------------------

    dim3 wmma_grid;
    dim3 wmma_block;
	wmma_grid.x = 68 * 2;
	wmma_block.x = THREADS_PER_BLOCK;
	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;

    printf("[ORI] Running with tzgemm...\n");
    printf("[ORI] wmma_grid -- %d * 1 wmma_block -- %d * 1 \n", wmma_grid.x, wmma_block.x);

    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((pers_tzgemm<<<wmma_grid, wmma_block>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C,
                            M_GLOBAL, N_GLOBAL, K_GLOBAL,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] tzgemm took %f ms\n", kernel_time);

    cublasHandle_t cublasHandle;
	cublasErrCheck(cublasCreate(&cublasHandle));
	cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
	printf("Running with cuBLAS...\n");

    cudaErrCheck(cudaEventRecord(startKERNEL));
	cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
					N_GLOBAL, M_GLOBAL, K_GLOBAL, 
					&alpha_g,
					mix_wmma_B, CUDA_R_16F, K_GLOBAL,
					mix_wmma_A, CUDA_R_16F, K_GLOBAL,
					&beta_g, 
					mix_wmma_C, CUDA_R_32F, N_GLOBAL,
					CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] tzgemm took %f ms\n", kernel_time);
    

    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(ori_result_C, ori_wmma_C, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(mix_result_C, mix_wmma_C, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
        float v1 = ori_result_C[i];
        float v2 = mix_result_C[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 20) printf("%f %f\n", v1, v2);
        }
		// if (i < 10) printf("%f %f\n", ori_result_C[i], mix_result_C[i]);
    }
    if (errors > 0) {
        printf("[WMMA] ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("[WMMA] Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }
    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));
    
    cudaErrCheck(cudaDeviceReset());
    return 0;
}
