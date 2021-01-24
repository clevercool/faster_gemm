#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <cublas_v2.h>

// helper functions and utilities to work with CUDA
// #include <helper_cuda.h>
#include "helper_cuda.h"
// #include <helper_functions.h>

// GPU configuration.
#define WARP_SIZE 32

// MMA matrix tile dimensions.


#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.
// #define M_TILES (32 * 8)
// #define N_TILES (4 * 48)
// #define K_TILES (4 * 12)
// #define M_GLOBAL (WMMA_M * M_TILES)
// #define N_GLOBAL (WMMA_N * N_TILES)
// #define K_GLOBAL (WMMA_K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global WMMA_K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4

#define CHUNK_LINE_BYTES (CHUNK_K * WMMA_K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

// Implementation constants.
#define WARPS_PER_BLOCK (BLOCK_ROW_WARPS * BLOCK_COL_WARPS)
// #define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define THREADS_PER_BLOCK 256

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL
#define SHMEM_STRIDE (WMMA_N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (WMMA_N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 8 two-byte
// "half" elements is chosen as the minimum possible shift because we must keep
// each row and column 128-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 16

const float alpha_g = 1.1f;
const float beta_g = 1.2f;

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

using namespace nvcuda;


__host__ void init_host_matrices(half *a, half *b, 
	int M_GLOBAL, int N_GLOBAL, int K_GLOBAL) {
	for (int i = 0; i < M_GLOBAL; i++) {
		for (int j = 0; j < K_GLOBAL; j++) {
			a[i * K_GLOBAL + j] = (half)(rand() % 3);
		}
	}

	for (int i = 0; i < N_GLOBAL; i++) {
		for (int j = 0; j < K_GLOBAL; j++) {
			b[i * K_GLOBAL + j] = (half)(rand() % 3);
		}
	}
}
		

__global__ void compute_gemm(const half *A, const half *B, const float *C,
							 float *D, float alpha, float beta, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL) {
  extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

  const unsigned int N_TILES = N_GLOBAL / N;
  const unsigned int K_TILES = K_GLOBAL / K;
  const unsigned int M_TILES = M_GLOBAL / M;

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C and D matrix tiles this warp computes.
  float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
                               (warpId / 2) * SHMEM_STRIDE * K * 2 +
                               (warpId % 2) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile to and
  // from shared memory.
  float *shmem_warp_stream_ptr =
      (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i =
        ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    const size_t gmem_idx =
        (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
    const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // Stream multiple C tiles to shared memory.
#pragma unroll
    for (int i = 0; i < K; i++) {
      typedef int4 copy_t;

      *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
          *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
            laneId);
    }

    __syncthreads();

    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES]
                                                       [WARP_ROW_TILES];

    // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        const float *tile_ptr =
            shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
          c[i][j].x[t] *= beta;
        }
      }
    }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
                                           M * K_GLOBAL * (warpId % 4) * 2)
                                        : (&B[block_tile_j * N * K_GLOBAL] +
                                           N * K_GLOBAL * (warpId % 4) * 2);

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
                                (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                       (laneId % CHUNK_COPY_LINE_LANES);

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
           i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr =
            (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>
            a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>
            b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
          const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
            }

            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }

      __syncthreads();
    }

      // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

        float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
          *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    __syncthreads();
  }
}


__global__ void pers_tzgemm(const half *A, const half *B, float *C, 
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		// float alpha, float beta,
		int grid_dimension_x, int block_dimension_x, int iteration) {

	__shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
	// extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

	unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

	// Warp and lane identification.
	const unsigned int warpId = thread_id_x / WARP_SIZE;
	const unsigned int laneId = thread_id_x % WARP_SIZE;

	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
	const unsigned int M_TILES = M_GLOBAL / WMMA_M;

	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;
	// This pointer is used to access the C and D matrix tiles this warp computes.
	float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
								(warpId / 2) * SHMEM_STRIDE * WMMA_M * 2 +
								(warpId % 2) * SHMEM_OFFSET;

	// This pointer is used to stream the C and D matrices block-wide tile to and
	// from shared memory.
	float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * WMMA_M;

	// Adjust the beta scaler, as it'll be multiplied by alpha at the end of
	// each tile computation. Technically this is not generally correct (may
	// result in a loss of precision). Zero still needs to be specially handled
	// though.
	float alpha = alpha_g;
	float beta = beta_g;
	beta /= alpha;

	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.
	for (;; block_pos += gridDim.x) {
		// if (block_pos >= grid_dimension_x) {
        //     return;
        // }

		const unsigned int block_tile_i =
			((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
		// Stop when there are no more D matrix tiles to compute in this CTA.
		if (block_tile_i >= M_TILES) {
			break;
		}
	  		// This warp's pointer to the C matrix data to copy memory from to shared
		// memory.
		const size_t gmem_idx =
			(block_tile_i + warpId) * WMMA_M * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_N;
			const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

		for (int loop = 0; loop < iteration; loop++) {
			
			// Stream multiple C tiles to shared memory.
		#pragma unroll
		for (int i = 0; i < 16; i++) {
		typedef int4 copy_t;

		*((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
			*((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
				laneId);
		}

		__syncthreads();

			// These fragments will accumulate the result of A and B matrix fragment
			// multiplications along the K_GLOBAL dimension.
			wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
			#pragma unroll
			for (int i = 0; i < WARP_COL_TILES; i++) {
				#pragma unroll
				for (int j = 0; j < WARP_ROW_TILES; j++) {
					wmma::fill_fragment(c[i][j], 0.0f);
				}
			}

			// Select what warp copies what matrix to shared memory.
			// Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
			const half *warp_ptr = 
				warpId < (WARPS_PER_BLOCK / 2) 
					? (&A[block_tile_i * WMMA_M * K_GLOBAL] + WMMA_M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
					: (&B[block_tile_j * WMMA_N * K_GLOBAL] + WMMA_N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2);

			// Go through the global K dimension by a fixed step at a time.
			#pragma unroll
			for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
				// Copy slices of the A and B matrices to shared memory.
				// The first half of the warps in the CTA copy the A matrix, 
				// the rest copy the B matrix.
				size_t shmem_idx =
					warpId < (WARPS_PER_BLOCK / 2)
						? (WMMA_M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
						: (WMMA_N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

				// First half of the warp copies the first row / column of the matrix,
				// the second half of the warp copies the next.
				int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * WMMA_K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) 
					+ (laneId % CHUNK_COPY_LINE_LANES);

				// Shift the second half of the warp to the next row / column in the
				// shared memory.
				shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

				#pragma unroll
				for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
					// Copy 16 bytes at once in each lane.
					*((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
						*lane_ptr;

					// Advance the global memory pointer and the shared memory index.
					lane_ptr =
						(int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
					shmem_idx += CHUNK_COPY_LINES_PER_WARP;
				}

				__syncthreads();

				// Compute a grid of C matrix tiles in each warp.
				#pragma unroll
				for (int k_step = 0; k_step < CHUNK_K; k_step++) {
					wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_COL_TILES];
					wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b[WARP_ROW_TILES];

					#pragma unroll
					for (int i = 0; i < WARP_COL_TILES; i++) {
						size_t shmem_idx_a = (warpId / 2) * WMMA_M * 2 + (i * WMMA_M);
						const half *tile_ptr = &shmem[shmem_idx_a][k_step * WMMA_K];
						wmma::load_matrix_sync(a[i], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);

						#pragma unroll
						for (int j = 0; j < WARP_ROW_TILES; j++) {
							if (i == 0) {
								// Load the B matrix fragment once, because it is going to be
								// reused against the other A matrix fragments.
								size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * WMMA_N) * (warpId % 2) + (j * WMMA_N);
								const half *tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_K];
								wmma::load_matrix_sync(b[j], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);
							}
							wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
						}
					}
				}
				__syncthreads();
			}

			// Store the D fragments to shared memory.
			#pragma unroll
			for (int i = 0; i < WARP_COL_TILES; i++) {
				#pragma unroll
				for (int j = 0; j < WARP_ROW_TILES; j++) {
					// Uniform, point-wise transformations of ALL fragment elements by ALL
					// threads in the warp are well-defined even though element indices
					// within fragment storage are not defined.
					#pragma unroll
					for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

					float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N;
					wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
				}
			}

			__syncthreads();

			// Now that shared memory contains all the D tiles, stream them to global
			// memory.
			float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

			#pragma unroll
			for (int i = 0; i < 16; i++) {
				*((int2 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
					*((int2 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
			}
			__syncthreads();
		}
	}
}


int main(int argc, char **argv) {
	int tzgemm_block_t = 0;
    int tzgemm_iter = 0;
	int M_INPUT = 0;
	int N_INPUT = 0;
	int K_INPUT = 0;
    if (argc == 6) {
        tzgemm_block_t = atoi(argv[1]);
        tzgemm_iter = atoi(argv[2]);
        M_INPUT = atoi(argv[3]);
        N_INPUT = atoi(argv[4]);
        K_INPUT = atoi(argv[5]);
    } else {
        tzgemm_block_t = 2;
        tzgemm_iter = 1000;
		M_INPUT = 256;
		N_INPUT = 256;
		K_INPUT = 256;
		// tzgemm_block_t = 2;
        // tzgemm_iter = 4000;
    }

	cudaDeviceProp deviceProp;
	cudaErrCheck(cudaGetDeviceProperties(&deviceProp, 0));

	// int M_TILES = 32 * 8;
	// int N_TILES = 4 * 48;
	// int K_TILES = 4 * 12;
	// int M_GLOBAL = WMMA_M * M_TILES;
	// int N_GLOBAL = WMMA_N * N_TILES;
	// int K_GLOBAL = WMMA_K * K_TILES;

	int M_GLOBAL = (M_INPUT < 64) ? 64 : (M_INPUT / 64) * 64;
	int N_GLOBAL = (N_INPUT < 64) ? 64 : (N_INPUT / 64) * 64;
	int K_GLOBAL = (K_INPUT < 64) ? 64 : (K_INPUT / 64) * 64;

	int M_TILES = M_GLOBAL / WMMA_M;
	int N_TILES = N_GLOBAL / WMMA_N;
	int K_TILES = K_GLOBAL / WMMA_K;

	printf("M_ORI: %5d M_GLOBAL: %5d (%d x %d) \n", M_INPUT, M_GLOBAL, WMMA_M, M_TILES);
	printf("N_ORI: %5d N_GLOBAL: %5d (%d x %d) \n", N_INPUT, N_GLOBAL, WMMA_N, N_TILES);
	printf("K_ORI: %5d K_GLOBAL: %5d (%d x %d) \n", K_INPUT, K_GLOBAL, WMMA_K, K_TILES);

	half *ori_host_A = NULL;
	half *ori_host_B = NULL;
	float *ori_result_C = NULL;
	float *cublas_result_C = NULL;

	half *ori_wmma_A = NULL;
	half *ori_wmma_B = NULL;
	float *ori_wmma_C = NULL;
	float *cublas_wmma_C = NULL;

	ori_host_A = (half *)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
	ori_host_B = (half *)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
	ori_result_C = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
	cublas_result_C = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

	init_host_matrices(ori_host_A, ori_host_B, M_GLOBAL, N_GLOBAL, K_GLOBAL);

	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_A), sizeof(half) * M_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_B), sizeof(half) * N_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_C), sizeof(float) * M_GLOBAL * N_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cublas_wmma_C), sizeof(float) * M_GLOBAL * N_GLOBAL));

	assert(((unsigned long long)ori_wmma_A) % 128 == 0);
	assert(((unsigned long long)ori_wmma_B) % 128 == 0);
	assert(((unsigned long long)ori_wmma_C) % 128 == 0);
	assert(((unsigned long long)cublas_wmma_C) % 128 == 0);

	// printf("Preparing data for GPU...\n");
	cudaErrCheck(cudaMemcpy(ori_wmma_A, ori_host_A, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
	cudaErrCheck(cudaMemcpy(ori_wmma_B, ori_host_B, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
	cudaErrCheck(cudaMemset(ori_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));
	cudaErrCheck(cudaMemset(cublas_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

	// enum {
	// 	// Compute the right amount of shared memory to request.
	// 	// We need shared memory to hold per-CTA C and D matrix tiles, 
	// 	// and to cache per-CTA chunks of the ori_wmma_A and ori_wmma_B matrices. 
	// 	// Therefore, the right amount to request is the maximum of those two numbers.
	// 	// sizeof(half) * (BLOCK_COL_TILES * WMMA_M) * (CHUNK_K * WMMA_K + SKEW_HALF) * 2
	// 	// 2 * (4 * 16) * (4 * 16 + 8) * 2 = 18 KB
	// 	// 2 * (8 * 16) * (4 * 16 + 8) * 2 = 36 KB
	// 	// WMMA_M * BLOCK_COL_TILES * WMMA_N * BLOCK_ROW_TILES * sizeof(float)
	// 	// 16 * 4 * 16 * 4 * 4 = 16 KB
	// 	// 16 * 8 * 16 * 8 * 4 = 64 KB
	// 	SHMEM_SZ = MAX(
	// 		sizeof(half) * (BLOCK_COL_TILES * WMMA_M) * (CHUNK_K * WMMA_K + SKEW_HALF) * 2,
	// 		WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
	// };
	// printf("Required shared memory size: %lu Kb\n\n", SHMEM_SZ / 1024UL);

	// const float alpha = 1.1f;
	// const float beta = 1.2f;

	enum {
		// Compute the right amount of shared memory to request.
		// We need shared memory to hold per-CTA C and D matrix tiles, and to cache
		// per-CTA chunks
		// of the A and B matrices. Therefore, the right amount to request is the
		// maximum of those
		// two numbers.
		SHMEM_SZ = MAX(
			sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
			M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
				(BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
	  };

	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaErrCheck(cudaEventCreate(&start));
	cudaErrCheck(cudaEventCreate(&stop));

	dim3 wmma_grid;
    dim3 wmma_block;
	wmma_grid.x = 80 * tzgemm_block_t;
	wmma_block.x = THREADS_PER_BLOCK;
	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;

	// If enough shared memory available on the GPU use high performant kernel
	printf("Running with tzgemm \n");
	cudaErrCheck(cudaEventRecord(start));
	// cudaErrCheck(cudaFuncSetAttribute(pers_tzgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
	// checkKernelErrors((pers_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C, alpha, beta)));
	// checkKernelErrors((pers_tzgemm<<<wmma_grid, wmma_block>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C, 
	// 						M_GLOBAL, N_GLOBAL, K_GLOBAL,
	// 						// alpha, beta,
	// 						wmma_grid_dim_x, wmma_block_dim_x, tzgemm_iter)));
	checkCudaErrors(cudaFuncSetAttribute(
        compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
    checkKernelErrors(
        (compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C, ori_wmma_C, alpha_g, beta_g, M_GLOBAL, N_GLOBAL, K_GLOBAL)));
						
	cudaErrCheck(cudaEventRecord(stop));
	cudaErrCheck(cudaEventSynchronize(stop));
	cudaErrCheck(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("tzgemm took %f ms\n", milliseconds);

	cudaErrCheck(cudaMemcpy(ori_result_C, ori_wmma_C, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

	cublasHandle_t cublasHandle;
	cublasErrCheck(cublasCreate(&cublasHandle));
	cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
	printf("Running with cuBLAS...\n");
	cudaErrCheck(cudaEventRecord(start));
	cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
				N_GLOBAL, M_GLOBAL, K_GLOBAL, 
				&alpha_g,
				ori_wmma_B, CUDA_R_16F, N_GLOBAL,
				ori_wmma_A, CUDA_R_16F, K_GLOBAL,
				&beta_g, 
				cublas_wmma_C, CUDA_R_32F, N_GLOBAL,
				CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
	cudaErrCheck(cudaEventRecord(stop));
	cudaErrCheck(cudaEventSynchronize(stop));
	cudaErrCheck(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("cublas took %f us\n", milliseconds * 1000);

	cudaErrCheck(cudaMemcpy(cublas_result_C, cublas_wmma_C, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

	printf("\nChecking results...\n");
	int count = 0;
	for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
		if (fabs(ori_result_C[i] - cublas_result_C[i]) > 0.1f) {
		count++;
		}

		// if (count < 10) {
		// 	printf("%f %f\n", ori_result_C[i], cublas_result_C[i]);
		// }
	}

	if (count > 0) {
		printf("[Errors] %d errors in %d numbers.\n", count, N_GLOBAL * M_GLOBAL);
	} else {
		printf("[Success]!!!\n");
	}
	free(ori_result_C);
	free(cublas_result_C);

	free(ori_host_A);
	free(ori_host_B);
	cudaErrCheck(cudaFree(reinterpret_cast<void *>(ori_wmma_A)));
	cudaErrCheck(cudaFree(reinterpret_cast<void *>(ori_wmma_B)));
	cudaErrCheck(cudaFree(reinterpret_cast<void *>(ori_wmma_C)));
	cudaErrCheck(cudaFree(reinterpret_cast<void *>(cublas_wmma_C)));

	return 0;
}
