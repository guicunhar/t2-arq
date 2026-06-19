NVCC=nvcc
CUDA_FLAGS=-lineinfo -lm

equation:	equation_test.cu gpu_lib.cu comum.h gpu.h
	$(NVCC) $(CUDA_FLAGS) -o equation equation_test.cu gpu_lib.cu