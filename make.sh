#2080ti
# nvcc -o tzgemm_out -arch=sm_75 -lcublas -lcurand --ptxas-options=-v -Xptxas -dlcm=ca faster_gemm.cu
# nvcc -o tzgemm_mix -arch=sm_75 -lcurand -lcublas --ptxas-options=-v -Xptxas -dlcm=ca tzgemm.cu
# nvcc -o sgemm_mix  -arch=sm_75 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_sgemm.cu
nvcc -o sgemm_stream  -arch=sm_75 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca stream_sgemm.cu
# nvcc -o cutcp_mix  -arch=sm_75 -lcurand --ptxas-options=-v -Xptxas -dlcm=ca mix_cutcp.cu
# V100
# nvcc -o  tzgemm_out -arch=sm_70 -lcublas faster_gemm.cu
