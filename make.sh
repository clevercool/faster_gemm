#2080ti
nvcc -o tzgemm_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca faster_gemm.cu
# V100
# nvcc -o  tzgemm_out -arch=sm_70 -lcublas faster_gemm.cu
