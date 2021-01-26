#define WMMA_GRID_DIM2 68 * 2
// #define WMMA_GRID_DIM2 gridDim.x

#define WARP_SIZE 32
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
#define BLOCK_COL_WARPS 2
#define WARP_ROW_TILES 2
#define WARP_COL_TILES 2

// Implementation constants.
#define WARPS_PER_BLOCK (BLOCK_ROW_WARPS * BLOCK_COL_WARPS)
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

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
#define SKEW_HALF 8

const float alpha_g = 1.1f;
const float beta_g = 1.2f;

// __host__ void init_host_matrices(half *a, half *b) {
// 	for (int i = 0; i < M_GLOBAL; i++) {
// 		for (int j = 0; j < K_GLOBAL; j++) {
// 			a[i * K_GLOBAL + j] = (half)(rand() % 3);
// 		}
// 	}

// 	for (int i = 0; i < N_GLOBAL; i++) {
// 		for (int j = 0; j < K_GLOBAL; j++) {
// 			b[i * K_GLOBAL + j] = (half)(rand() % 3);
// 		}
// 	}
// }


__global__ void pers_tzgemm(half *A, half *B, float *C, 
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		int grid_dimension_x, int block_dimension_x, int iteration) {

	__shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
	// extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
	const unsigned int M_TILES = M_GLOBAL / WMMA_M;

	float alpha = alpha_g;
	float beta = beta_g;

	unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

	// Warp and lane identification.
	const unsigned int warpId = thread_id_x / WARP_SIZE;
	const unsigned int laneId = thread_id_x % WARP_SIZE;

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
	beta /= alpha;

	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.
	for (;; block_pos += gridDim.x) {
		if (block_pos >= grid_dimension_x) {
            return;
        }

		const unsigned int block_tile_i =
			((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
		// This warp's pointer to the C matrix data to copy memory from to shared
		// memory.
		const size_t gmem_idx =
			(block_tile_i + warpId) * WMMA_M * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_N;

		for (int loop = 0; loop < iteration; loop++) {

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


__device__ void mix_tzgemm(half *A, half *B, float *C, 
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		int grid_dimension_x, int block_dimension_x, int iteration) {

	__shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
	// extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
	const unsigned int M_TILES = M_GLOBAL / WMMA_M;

	float alpha = alpha_g;
	float beta = beta_g;

	unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

	// Warp and lane identification.
	const unsigned int warpId = thread_id_x / WARP_SIZE;
	const unsigned int laneId = thread_id_x % WARP_SIZE;

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
	beta /= alpha;

	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.
	for (;; block_pos += WMMA_GRID_DIM2) {
		if (block_pos >= grid_dimension_x) {
            return;
        }

		const unsigned int block_tile_i =
			((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
		// This warp's pointer to the C matrix data to copy memory from to shared
		// memory.
		const size_t gmem_idx =
			(block_tile_i + warpId) * WMMA_M * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_N;

		for (int loop = 0; loop < iteration; loop++) {

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

				// __syncthreads();
				asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");

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
				// __syncthreads();
				asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");
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

			// __syncthreads();
			asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");

			// Now that shared memory contains all the D tiles, stream them to global
			// memory.
			float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

			#pragma unroll
			for (int i = 0; i < 16; i++) {
				*((int2 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
					*((int2 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
			}
			// __syncthreads();
			asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(128) : "memory");
		}
	}
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}