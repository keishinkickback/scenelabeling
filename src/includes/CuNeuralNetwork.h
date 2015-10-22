/*
 * CuNeuralNetwork.h
 *
 *  Created on: Aug 17, 2015
 *      Author: ryukinkou
 */

#ifndef CUNEURALNETWORK_H_
#define CUNEURALNETWORK_H_

#include <cuda_runtime.h>
#include <cudnn.h>

struct cudnnOutputDim {

public:

	int outputImages;
	int outputFeaturemapsForEachImage;
	int outputFeaturemapHeight;
	int outputFeaturemapWidth;
};

class CuNeuralNetwork {

public:

	float * initializeInputDataLayer(float * h_input_data,
			cudnnTensorDescriptor_t * inputDataTensorDescriptor, int batchSize,
			int inputFeaturemaps, int imageHeight, int imageWidth);

	float * initializeKernels(float * h_kernel,
			cudnnFilterDescriptor_t * kernelDescriptor, int inputFeaturemaps,
			int outputFeaturemaps, int kernelHeight, int kernelWidth);

	float * initializeOutputDataLayer(
			cudnnTensorDescriptor_t * inputDataTensorDescriptor,
			cudnnFilterDescriptor_t * kernelDescriptor,
			cudnnConvolutionDescriptor_t * convolutionDescriptor,
			cudnnTensorDescriptor_t * outputDataTensorDescriptor,
			cudnnOutputDim * outputDim);

	float * addBiasUnits(float * h_bias,
			cudnnTensorDescriptor_t * biasTensorDescriptor,
			int outputFeaturemaps, int kernelHeight, int kernelWidth);

	float * initializePoolingLayer(float * d_output_data,
			cudnnTensorDescriptor_t * inputDataTensorDescriptor,
			cudnnPoolingDescriptor_t * poolingDescriptor,
			cudnnTensorDescriptor_t * poolingDataTensorDescriptor,
			cudnnOutputDim * outputDim, int poolingWindowHeight,
			int poolingWindowWidth, int poolingVerticalStride,
			int poolingHorizontalStride, cudnnOutputDim * poolingOutputDim);

};

#endif /* CUNEURALNETWORK_H_ */
