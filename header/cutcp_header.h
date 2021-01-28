
#define BIN_DEPTH         8  /* max number of atoms per bin */
#define BIN_SIZE         32  /* size of bin in floats */
#define BIN_SHIFT         5  /* # of bits to shift for mul/div by BIN_SIZE */
#define BIN_CACHE_MAXLEN 32  /* max number of atom bins to cache */

#define BIN_LENGTH      4.f  /* spatial length in Angstroms */
#define BIN_INVLEN  (1.f / BIN_LENGTH)
/* assuming density of 1 atom / 10 A^3, expectation is 6.4 atoms per bin
 * so that bin fill should be 80% (for non-empty regions of space) */
#define REGION_SIZE     512  /* number of floats in lattice region */

#define NBRLIST_DIM  11
#define NBRLIST_MAXLEN (NBRLIST_DIM * NBRLIST_DIM * NBRLIST_DIM)

__constant__ int NbrListLen;
__constant__ int3 NbrList[NBRLIST_MAXLEN];

#define CUTCP_GRID_DIM gridDim.x
// #define CUTCP_GRID_DIM gridDim.x
