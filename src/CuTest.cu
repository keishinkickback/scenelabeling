#include <iostream>
#include <sstream>
#include <numeric>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <FreeImage.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>

#include "includes/CuTest.h"

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

//int main(void) {
//	float alpha = 1.0;
//	float beta = 0.0;
//	float h_A[6] = { 1, 1, 2, 2, 3, 3 };
//	float h_B[2] = { 1, 1 };
//	float h_C[3];
//	float *d_a, *d_b, *d_c;
//	checkCudaErrors(cudaMalloc((void** ) &d_a, 6 * sizeof(float)));
//	checkCudaErrors(cudaMalloc((void** ) &d_b, 2 * sizeof(float)));
//	checkCudaErrors(cudaMalloc((void** ) &d_c, 3 * sizeof(float)));
//	checkCudaErrors(
//			cudaMemcpy(d_a, &h_A, 6 * sizeof(float), cudaMemcpyHostToDevice));
//	checkCudaErrors(
//			cudaMemcpy(d_b, &h_B, 2 * sizeof(float), cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemset(d_c, 0, 3 * sizeof(float)));
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//
//	CuNeuralNetwork network;
//
//	cublasSgemm(
//			handle,
//			CUBLAS_OP_N,
//			CUBLAS_OP_N,
//			1,
//			3,
//			2,
//			&alpha,
//			d_b,
//			1,
//			d_a,
//			2,
//			&beta,
//			d_c,
//			1);
//
//
//	checkCudaErrors(
//			cudaMemcpy(h_C, d_c, 3 * sizeof(float), cudaMemcpyDeviceToHost));
//	for (int i = 0; i < 3; i++) {
//		printf("%f\n", h_C[i]);
//	}
//	printf("\n");
//	return 0;
//}
