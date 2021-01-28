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
#include "header/cutcp_header.h"

#include "file_t/tzgemm_kernel.cu"
#include "file_t/cutcp_kernel.cu"


__global__ void mix_kernel(
    half *a, half *b, float *c,
    int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    int binDim_x,
    int binDim_y,
    float4 *binZeroAddr,    /* address of atom bins starting at origin */
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    float *regionZeroAddr, /* address of lattice regions starting at origin */
    int zRegionIndex_t,
    int cutcp_grid_dim_x,
    int cutcp_grid_dim_y,
    int cutcp_grid_dim_z,
    int cutcp_block_dim_x,
    int cutcp_block_dim_y,
    int cutcp_block_dim_z,
    int cutcp_iter){
    if (threadIdx.x < wmma_block_dim_x * 1 && blockIdx.x < WMMA_GRID_DIM2) {
        mix_tzgemm(a, b, c, 
        MATRIX_M, MATRIX_N, MATRIX_K,
        wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x >= wmma_block_dim_x * 1 && blockIdx.x < CUTCP_GRID_DIM) {
        int thread_step = wmma_block_dim_x * 1;
        mix_cutcp(
            binDim_x,
            binDim_y,
            binZeroAddr,    /* address of atom bins starting at origin */
            h,                /* lattice spacing */
            cutoff2,          /* square of cutoff distance */
            inv_cutoff2,
            regionZeroAddr, /* address of lattice regions starting at origin */
            zRegionIndex_t,
            cutcp_grid_dim_x,
            cutcp_grid_dim_y,
            cutcp_grid_dim_z,
            cutcp_block_dim_x,
            cutcp_block_dim_y,
            cutcp_block_dim_z,
            thread_step,
            cutcp_iter
            );
    }
}



int main(int argc, char* argv[]) {
    int cutcp_blks = 2;
    int cutcp_iter = 33;
	int wmma_blks = 2;
    int wmma_iter = 28000;
    int M_INPUT = 128;
	int N_INPUT = 3136;
	int K_INPUT = 1152;
	int mixwarp = 2;
    if (argc == 6) {
        cutcp_blks = atoi(argv[1]);
        cutcp_iter = atoi(argv[2]);
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

    // tcgemm variables
    // ---------------------------------------------------------------------------------------
    int MATRIX_M = (M_INPUT < 64) ? 64 : (M_INPUT / 64) * 64;
	int MATRIX_N = (N_INPUT < 64) ? 64 : (N_INPUT / 64) * 64;
	int MATRIX_K = (K_INPUT < 64) ? 64 : (K_INPUT / 64) * 64;

	int M_TILES = MATRIX_M / WMMA_M;
	int N_TILES = MATRIX_N / WMMA_N;
	int K_TILES = MATRIX_K / WMMA_K;

	printf("M_ORI: %5d MATRIX_M: %5d (%d x %d) \n", M_INPUT, MATRIX_M, WMMA_M, M_TILES);
	printf("N_ORI: %5d MATRIX_N: %5d (%d x %d) \n", N_INPUT, MATRIX_N, WMMA_N, N_TILES);
	printf("K_ORI: %5d MATRIX_K: %5d (%d x %d) \n", K_INPUT, MATRIX_K, WMMA_K, K_TILES);

    float *ori_host_A = NULL;
	float *ori_host_B = NULL;
	float *host_wmma_ori_c = NULL;
	float *host_wmma_pers_c = NULL;

	half *wmma_ori_a = NULL;
	half *wmma_ori_b = NULL;
	float *wmma_ori_c = NULL;
	float *wmma_pers_c = NULL;

	host_wmma_ori_c = (float *)malloc(sizeof(float) * MATRIX_M * MATRIX_N);
	host_wmma_pers_c = (float *)malloc(sizeof(float) * MATRIX_M * MATRIX_N);

	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_A), sizeof(float) * MATRIX_M * MATRIX_K));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_B), sizeof(float) * MATRIX_N * MATRIX_K));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_a), sizeof(half) * MATRIX_M * MATRIX_K));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_b), sizeof(half) * MATRIX_N * MATRIX_K));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_c), sizeof(float) * MATRIX_M * MATRIX_N));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_pers_c), sizeof(float) * MATRIX_M * MATRIX_N));

	assert(((unsigned long long)wmma_ori_a) % 128 == 0);
	assert(((unsigned long long)wmma_ori_b) % 128 == 0);
	assert(((unsigned long long)wmma_ori_c) % 128 == 0);
	assert(((unsigned long long)wmma_pers_c) % 128 == 0);

	curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
	curandErrCheck(curandGenerateUniform(gen, ori_host_A, MATRIX_M * MATRIX_K));
    curandErrCheck(curandGenerateUniform(gen, ori_host_B, MATRIX_N * MATRIX_K));
	convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_a, ori_host_A, MATRIX_M * MATRIX_K);
    convertFp32ToFp16 <<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_b, ori_host_B, MATRIX_N * MATRIX_K);
	cudaErrCheck(cudaMemset(wmma_ori_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));
	cudaErrCheck(cudaMemset(wmma_pers_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));

    // cutcp variables
    // ---------------------------------------------------------------------------------------
    int nbrlistlen = 335;
    // int3 *nbrlist;
    int3 nbrlist[NBRLIST_MAXLEN];

    float *cutcp_ori_regionZeroCuda;
    float4 *cutcp_ori_binBaseCuda, *cutcp_ori_binZeroCuda, *host_cutcp_ori_binBaseCuda;
    float *cutcp_pers_regionZeroCuda;
    float4 *cutcp_pers_binBaseCuda, *cutcp_pers_binZeroCuda, *host_cutcp_pers_binBaseCuda;

    int lnx = 208;
    int lny = 208;
    int lnz = 208;
    int lnall = lnx * lny * lnz;
    int nbins = 32768;

    int zRegionDim = 26;
    int binDim_x = 32;
    int binDim_y = 32;
    float h = 0.5;
    float cutoff2 = 144.0;
    float inv_cutoff2 = 0.006944;

    // nbrlist = (int3 *)malloc(NBRLIST_MAXLEN * sizeof(int3));
    // memset(nbrlist, 1, NBRLIST_MAXLEN * sizeof(int3));
    for (int i = 0; i < NBRLIST_MAXLEN; i++){
        nbrlist[i].x = i % 7 - 3;
        nbrlist[i].y = i % 7 - 3;
        nbrlist[i].z = i % 7 - 3;
    }

    cudaErrCheck(cudaMalloc((void **) &cutcp_ori_regionZeroCuda, lnall * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **) &cutcp_pers_regionZeroCuda, lnall * sizeof(float)));
    cudaErrCheck(cudaMemset(cutcp_ori_regionZeroCuda, 2, lnall * sizeof(float)));
    cudaErrCheck(cudaMemset(cutcp_pers_regionZeroCuda, 2, lnall * sizeof(float)));

    cudaErrCheck(cudaMalloc((void **) &cutcp_ori_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4)));
    cudaErrCheck(cudaMalloc((void **) &cutcp_pers_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4)));
    cudaErrCheck(cudaMemset(cutcp_ori_binBaseCuda, 3, nbins * BIN_DEPTH * sizeof(float4)));
    cudaErrCheck(cudaMemset(cutcp_pers_binBaseCuda, 3, nbins * BIN_DEPTH * sizeof(float4)));
    cutcp_ori_binZeroCuda = cutcp_ori_binBaseCuda + ((3 * binDim_y + 3) * binDim_x + 3) * BIN_DEPTH;
    cutcp_pers_binZeroCuda = cutcp_pers_binBaseCuda + ((3 * binDim_y + 3) * binDim_x + 3) * BIN_DEPTH;

    host_cutcp_ori_binBaseCuda = (float4 *)malloc(nbins * BIN_DEPTH * sizeof(float4));
    host_cutcp_pers_binBaseCuda = (float4 *)malloc(nbins * BIN_DEPTH * sizeof(float4));

    cudaErrCheck(cudaMemcpyToSymbol(NbrListLen, &nbrlistlen, sizeof(int), 0));
	cudaErrCheck(cudaMemcpyToSymbol(NbrList, nbrlist, nbrlistlen * sizeof(int3), 0));


    // SOLO running
    // ---------------------------------------------------------------------------------------
    dim3 wmma_grid;
    dim3 wmma_block;
	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	wmma_block.x = THREADS_PER_BLOCK;

	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;
	wmma_grid.x = 68 * wmma_blks;
	wmma_block.x = THREADS_PER_BLOCK;

    printf("[PERS] Running with tzgemm...\n");
    printf("[PERS] wmma_grid -- %d * %d wmma_block -- %d * %d \n", wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);

	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((pers_tzgemm<<<wmma_grid, wmma_block>>>(wmma_ori_a, wmma_ori_b, wmma_pers_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
	printf("[PERS] tzgemm took %f ms\n", kernel_time);
    serial_time += kernel_time;

	// SOLO running
    // ---------------------------------------------------------------------------------------
    dim3 cutcp_grid, cutcp_block;
    cutcp_grid.x = zRegionDim;
	cutcp_grid.y = zRegionDim;
	cutcp_grid.z = zRegionDim;
	cutcp_block.x = 8;
	cutcp_block.y = 2;
	cutcp_block.z = 8;

    // printf("[ORI] Running with cutcp...\n");
    // printf("[ORI] cutcp_grid -- %d * %d * %d cutcp_block -- %d * %d * %d \n", cutcp_grid.x, cutcp_grid.y, cutcp_grid.z, cutcp_block.x, cutcp_block.y, cutcp_block.z);
    // // int num_blocks = 0;
    // // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, ori_cutcp, cutcp_block.x * cutcp_block.y * cutcp_block.z, 0);
    // // printf("[cutcp] cudaOccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);
    // cudaErrCheck(cudaEventRecord(startKERNEL));
    // checkKernelErrors((ori_cutcp<<<cutcp_grid, cutcp_block>>>(binDim_x, binDim_y, cutcp_ori_binZeroCuda, h, cutoff2, inv_cutoff2, cutcp_ori_regionZeroCuda, 25, cutcp_iter)));
    // cudaErrCheck(cudaEventRecord(stopKERNEL));
    // cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    // cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    // printf("[ORI] cutcp took %f ms\n\n", kernel_time);

    // PTB
    // ---------------------------------------------------------------------------------
    int cutcp_grid_dim_x = cutcp_grid.x;
    int cutcp_grid_dim_y = cutcp_grid.y;
    int cutcp_grid_dim_z = cutcp_grid.z;
    int cutcp_block_dim_x = cutcp_block.x;
    int cutcp_block_dim_y = cutcp_block.y;
    int cutcp_block_dim_z = cutcp_block.z;
    cutcp_grid.x = cutcp_blks == 0 ? cutcp_grid_dim_x * cutcp_grid_dim_y * cutcp_grid_dim_z : 68 * cutcp_blks;
    cutcp_grid.y = 1;
    cutcp_grid.z = 1;
    cutcp_block.x = cutcp_block_dim_x * cutcp_block_dim_y * cutcp_block_dim_z;
    cutcp_block.y = 1;
    cutcp_block.z = 1;

    cudaErrCheck(cudaMemcpyToSymbol(NbrListLen, &nbrlistlen, sizeof(int), 0));
	cudaErrCheck(cudaMemcpyToSymbol(NbrList, nbrlist, nbrlistlen * sizeof(int3), 0));

    printf("[PERS] Running with cutcp...\n");
    printf("[PERS] cutcp_grid -- %d * %d * %d cutcp_block -- %d * %d * %d \n", cutcp_grid.x, cutcp_grid.y, cutcp_grid.z, cutcp_block.x, cutcp_block.y, cutcp_block.z);
    // num_blocks = 0;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, pers_cutcp, cutcp_block.x * cutcp_block.y * cutcp_block.z, 0);
    // printf("[PERS] cudaOccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);
    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((pers_cutcp<<<cutcp_grid, cutcp_block>>>(binDim_x, binDim_y, cutcp_pers_binZeroCuda, 
        h, cutoff2, inv_cutoff2, cutcp_pers_regionZeroCuda, 25, cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z,
        cutcp_block_dim_x, cutcp_block_dim_y, cutcp_block_dim_z, cutcp_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[PERS] cutcp took %f ms\n\n", kernel_time);
    serial_time += kernel_time;

	// MIX running 
    // ----------------------------------------------------------------------------------------------------------------------
    cudaErrCheck(cudaMemcpyToSymbol(NbrListLen, &nbrlistlen, sizeof(int), 0));
	cudaErrCheck(cudaMemcpyToSymbol(NbrList, nbrlist, nbrlistlen * sizeof(int3), 0));
	if (mixwarp == 1) {
		dim3 mix_grid, mix_block;
        mix_grid.x = (cutcp_grid.x > wmma_grid.x) ? cutcp_grid.x : wmma_grid.x;
        mix_grid.y = 1;
        mix_block.x = cutcp_block.x + wmma_block.x;
        mix_block.y = 1;
		printf("[MIX] mix_grid -- %d * %d mix_block -- %d * %d \n", mix_grid.x, mix_grid.y, mix_block.x, mix_block.y);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((mix_kernel <<<mix_grid, mix_block>>> (
			// wmma parameters
			wmma_ori_a, wmma_ori_b, wmma_ori_c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter,
			// sgemm parameters
			binDim_x, binDim_y, cutcp_ori_binZeroCuda, 
            h, cutoff2, inv_cutoff2, cutcp_ori_regionZeroCuda, 25, cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z,
            cutcp_block_dim_x, cutcp_block_dim_y, cutcp_block_dim_z, cutcp_iter
		)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PETS] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 2) {
		cudaErrCheck(cudaEventRecord(startKERNEL));
		
		
        checkKernelErrors((pers_tzgemm<<<wmma_grid, wmma_block, 0, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
        checkKernelErrors((pers_cutcp<<<cutcp_grid, cutcp_block, 0, streams[1]>>>(binDim_x, binDim_y, cutcp_ori_binZeroCuda, 
            h, cutoff2, inv_cutoff2, cutcp_ori_regionZeroCuda, 25, cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z,
            cutcp_block_dim_x, cutcp_block_dim_y, cutcp_block_dim_z, cutcp_iter)));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[STREAMP] mix took %f ms\n\n", kernel_time);
	} else {
        cutcp_grid.x = zRegionDim;
        cutcp_grid.y = zRegionDim;
        cutcp_grid.z = zRegionDim;
        cutcp_block.x = 8;
        cutcp_block.y = 2;
        cutcp_block.z = 8;
        cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((pers_tzgemm<<<wmma_grid, wmma_block, 0, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		checkKernelErrors((ori_cutcp<<<cutcp_grid, cutcp_block, 0, streams[1]>>>(binDim_x, binDim_y, cutcp_ori_binZeroCuda, 
            h, cutoff2, inv_cutoff2, cutcp_ori_regionZeroCuda, 25, cutcp_iter)));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[STREAMO] mix took %f ms\n\n", kernel_time);
    }

    printf("[STAT] Overlap rate: %.2f\n", (serial_time - kernel_time) * 100 / serial_time);
    printf("[STAT] Throughput speedup: %.2f\n", (serial_time / kernel_time - 1) * 100);

	// Checking results
    // ---------------------------------------------------------------------------------------
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(host_wmma_ori_c, wmma_ori_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_wmma_pers_c, wmma_pers_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_cutcp_ori_binBaseCuda, cutcp_ori_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_cutcp_pers_binBaseCuda, cutcp_pers_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
        float v1 = host_wmma_ori_c[i];
        float v2 = host_wmma_pers_c[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
		if (i < 3) printf("%d %f %f\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("[WMMA] ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("[WMMA] Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }
    errors = 0;
    for (int i = 0; i < nbins * BIN_DEPTH; i++) {
        float v1 = host_cutcp_ori_binBaseCuda[i].x;
        float v2 = host_cutcp_pers_binBaseCuda[i].x;
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
        v1 = host_cutcp_ori_binBaseCuda[i].y;
        v2 = host_cutcp_pers_binBaseCuda[i].y;
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
        v1 = host_cutcp_ori_binBaseCuda[i].z;
        v2 = host_cutcp_pers_binBaseCuda[i].z;
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
        v1 = host_cutcp_ori_binBaseCuda[i].w;
        v2 = host_cutcp_pers_binBaseCuda[i].w;
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
    }
    if (errors > 0) {
        printf("ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }

    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaDeviceReset());
    return 0;
}