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

class CuNeuralNetwork {

public:

	void initializeConvolutionalLayerTensorDescriptor(
			cudnnHandle_t *cudnnHandle,
			cudnnTensorDescriptor_t * inputDataTensorDescriptor,
			cudnnFilterDescriptor_t * kernelDescriptor,
			cudnnConvolutionDescriptor_t * convolutionDescriptor,
			cudnnTensorDescriptor_t * outputDataTensorDescriptor,
			cudnnConvolutionFwdAlgo_t * algorithm, int executeBatchSize,
			int imageHeight, int imageWidth, int kernelHeight, int kernelWidth,
			int inputFeaturemaps, int outputFeaturemaps,
			size_t * workspaceSizeInByte, int * outputImages,
			int * outputFeaturemapsForEachImage, int * outputFeaturemapHeight,
			int * outputFeaturemapWidth);

	void syncTrainingDataToDevice(float * h_input_data, float * d_input_data,
			float * h_kernel, float * d_kernel, float * d_output_data,
			void * d_cudnn_workspace, int executeBatchSize, int imageHeight,
			int imageWidth, int kernelHeight, int kernelWidth,
			int inputFeaturemaps, int outputFeaturemaps, int outputImages,
			int outputFeaturemapsForEachImage, int outputFeaturemapHeight,
			int outputFeaturemapWidth, size_t workspaceSizeInByte);
};

#endif /* CUNEURALNETWORK_H_ */
