#include <stdio.h>
#include <assert.h>
#include <curand.h>
#include <cublas_v2.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)


#include <mma.h>
using namespace nvcuda; 

#include "header/tzgemm_header.h"
#include "header/sgemm_header.h"

#include "file_t/tzgemm_kernel.cu"
#include "file_t/sgemm_kernel.cu"


// __global__ void mix_kernel(
//     half *a, half *b, float *c,
// 	int MATRIX_M, int MATRIX_N, int MATRIX_K,
//     int wmma_grid_dim_x, int wmma_block_dim_x, 
//     int wmma_iter,
//     float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
//     int sgemm_grid_dim_x, int sgemm_grid_dim_y, int sgemm_block_dim_x, int sgemm_block_dim_y,
//     int sgemm_iter){
//     if (threadIdx.x < wmma_block_dim_x * 1 && blockIdx.x < WMMA_GRID_DIM2) {
//         mix_tzgemm(a, b, c, MATRIX_M, MATRIX_N, MATRIX_K,
// 		wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
//     } else if (threadIdx.x >= wmma_block_dim_x * 1 && blockIdx.x < SGEMM_GRID_DIM) {
//         int thread_step = wmma_block_dim_x * 1;
//         mix_sgemm(A, B, C, NORMAL_M, NORMAL_N, NORMAL_K, 
//                 sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter, thread_step);
//     }
// }


__global__ void mix_kernel(
    half *a, half *b, float *c,
	int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
    int sgemm_grid_dim_x, int sgemm_grid_dim_y, int sgemm_block_dim_x, int sgemm_block_dim_y,
    int sgemm_iter){
    if (threadIdx.x < 128 && blockIdx.x < 68 * 2) {
        mix_sgemm(A, B, C, NORMAL_M, NORMAL_N, NORMAL_K, 
                sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter, 0);
    } else if (threadIdx.x < 128 && blockIdx.x >= 68 * 2) {
        mixx_tzgemm(a, b, c, MATRIX_M, MATRIX_N, MATRIX_K,
		wmma_grid_dim_x, wmma_block_dim_x, wmma_iter, 68 * 2);
    }
}


int main(int argc, char* argv[]) {
    int sgemm_blks = 2;
    int sgemm_iter = 33;
	int wmma_blks = 2;
    // int wmma_iter = 1700;
    // int M_INPUT = 64 * 64;
	// int N_INPUT = 64 * 48;
	// int K_INPUT = 64 * 12;
    int wmma_iter = 28000;
    int M_INPUT = 128;
	int N_INPUT = 3136;
	int K_INPUT = 1152;
	int mixwarp = 2;
    if (argc == 6) {
        sgemm_blks = atoi(argv[1]);
        sgemm_iter = atoi(argv[2]);
        wmma_blks = atoi(argv[3]);
        wmma_iter = atoi(argv[4]);
		mixwarp = atoi(argv[5]);
    }

    // variables
    // ---------------------------------------------------------------------------------------
    float kernel_time;
    float serial_time = 0;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));
	cudaStream_t streams[2];
    for (int i = 0; i < 2; i++) {
        cudaErrCheck(cudaStreamCreate(&streams[i]));
    }

    // sgemm variables
    // ---------------------------------------------------------------------------------------
    float *sgemm_ori_a;
    float *sgemm_ori_b;
    float *sgemm_ori_c;
    float *sgemm_pers_a;
    float *sgemm_pers_b;
    float *sgemm_pers_c;
    float *host_sgemm_ori_c;
    float *host_sgemm_pers_c;

    int NORMAL_M = 4096;
    int NORMAL_N = 4128;
    int NORMAL_K = 4064;

    cudaErrCheck(cudaMalloc((void**)&sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_pers_a, NORMAL_M * NORMAL_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_pers_b, NORMAL_K * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_pers_c, NORMAL_M * NORMAL_N * sizeof(float)));

    host_sgemm_ori_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
    host_sgemm_pers_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));

    curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, sgemm_ori_a, NORMAL_M * NORMAL_K));
    curandErrCheck(curandGenerateUniform(gen, sgemm_ori_b, NORMAL_K * NORMAL_N));
    cudaErrCheck(cudaMemcpy(sgemm_pers_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(sgemm_pers_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
    curandErrCheck(curandDestroyGenerator(gen));
	cudaErrCheck(cudaMemset(sgemm_ori_c, 0, sizeof(float) * NORMAL_M * NORMAL_N));
	cudaErrCheck(cudaMemset(sgemm_pers_c, 0, sizeof(float) * NORMAL_M * NORMAL_N));


	// SOLO running
    // ---------------------------------------------------------------------------------------
    dim3 sgemm_grid;
    dim3 sgemm_block;
    sgemm_block.x = TILE_N;
    sgemm_block.y = TILE_TB_HEIGHT;
    sgemm_grid.x = NORMAL_M/TILE_M;
    sgemm_grid.y = NORMAL_N/TILE_N;
    printf("[ORI] Running with sgemm...\n");
    printf("[ORI] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);
    // int num_blocks = 0;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, ori_sgemm, sgemm_block.x * sgemm_block.y, 0);
    // printf("[cutcp] cudaOccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);

    // int sgemm_grid_dim_x = sgemm_grid.x;
    // int sgemm_grid_dim_y = sgemm_grid.y;
    // int sgemm_block_dim_x = sgemm_block.x;
    // int sgemm_block_dim_y = sgemm_block.y;
    // sgemm_grid.x = sgemm_grid_dim_x * sgemm_grid_dim_y;
    // sgemm_grid.x = sgemm_blks == 0 ? sgemm_grid_dim_x * sgemm_grid_dim_y : 68 * sgemm_blks;
    // sgemm_grid.y = 1;
    // sgemm_block.x = sgemm_block_dim_x * sgemm_block_dim_y;
    // sgemm_block.y = 1;

    cudaErrCheck(cudaEventRecord(startKERNEL));

    // checkKernelErrors((pers_sgemm <<< sgemm_grid, sgemm_block, 0, streams[0] >>> (sgemm_ori_a, sgemm_ori_b, sgemm_ori_c, 
    //             NORMAL_M, NORMAL_N, NORMAL_K,
    //             sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter / 1)));
    checkKernelErrors((ori_sgemm <<< sgemm_grid, sgemm_block >>> (sgemm_ori_a, sgemm_ori_b, sgemm_ori_c, 
                            NORMAL_M, NORMAL_N, NORMAL_K, sgemm_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] sgemm took %f ms\n\n", kernel_time);
    serial_time = kernel_time * 2;

	cudaErrCheck(cudaMemset(sgemm_ori_c, 0, sizeof(float) * NORMAL_M * NORMAL_N));

	cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ori_sgemm <<< sgemm_grid, sgemm_block, 0, streams[0] >>> (sgemm_ori_a, sgemm_ori_b, sgemm_ori_c, 
			NORMAL_M, NORMAL_N, NORMAL_K,
			sgemm_iter)));
    checkKernelErrors((ori_sgemm <<< sgemm_grid, sgemm_block, 0, streams[1] >>> (sgemm_pers_a, sgemm_pers_b, sgemm_pers_c, 
			NORMAL_M, NORMAL_N, NORMAL_K,
			sgemm_iter)));

    // checkKernelErrors((pers_sgemm <<< sgemm_grid, sgemm_block, 0, streams[0] >>> (sgemm_ori_a, sgemm_ori_b, sgemm_ori_c, 
    //             NORMAL_M, NORMAL_N, NORMAL_K,
    //             sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter)));
    // checkKernelErrors((pers_sgemm <<< sgemm_grid, sgemm_block, 0, streams[1] >>> (sgemm_pers_a, sgemm_pers_b, sgemm_pers_c, 
    //             NORMAL_M, NORMAL_N, NORMAL_K,
    //             sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[STREAM ORI] mix took %f ms\n\n", kernel_time);


    printf("[STAT] Overlap rate: %.2f\n", (serial_time - kernel_time) * 100 / serial_time);
    printf("[STAT] Throughput speedup: %.2f\n", (serial_time / kernel_time - 1) * 100);

	// Checking results
    // ---------------------------------------------------------------------------------------
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(host_sgemm_ori_c, sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_sgemm_pers_c, sgemm_pers_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < NORMAL_M * NORMAL_N; i++) {
        float v1 = host_sgemm_ori_c[i];
        float v2 = host_sgemm_pers_c[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
		if (i < 3) printf("%d %f %f\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("[SGEMM] ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("[SGEMM] Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }

    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaDeviceReset());
    return 0;
}