/*
 ============================================================================
 Name        : scenelabeling.cu
 Author      : liujinhang @ whut
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

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

#include "CuNeuralNetwork.h"

#include <vector>

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

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

class ImageProcessor {

public:
	void readRGBImage(char *imagePath, std::vector<float> *redChannel,
			std::vector<float> *greenChannel, std::vector<float> *blueChannel) {

		FreeImage_Initialise(TRUE);

		FIBITMAP* fib;
		fib = FreeImage_Load(FIF_PNG, imagePath, PNG_DEFAULT);
		int width = FreeImage_GetWidth(fib);
		int height = FreeImage_GetHeight(fib);

		RGBQUAD color;

		for (int x = 0; x < width; x++) {

			for (int y = 0; y < height; y++) {

				FreeImage_GetPixelColor(fib, x, y, &color);

				float blue = color.rgbBlue;
				float green = color.rgbGreen;
				float red = color.rgbRed;
				redChannel->push_back(red);
				greenChannel->push_back(green);
				blueChannel->push_back(blue);

			}

		}

		FreeImage_Unload(fib);
		FreeImage_DeInitialise();
	}

	std::vector<float> imageChannelNormalization(std::vector<float> *channel) {

		float maxColorChannel = *std::max_element(channel->begin(),
				channel->end());
		float minColorChannel = *std::min_element(channel->begin(),
				channel->end());

		std::vector<float> result;

		for (int i = 0; i < channel->size(); i++) {
			result.push_back(
					(channel->at(i) - minColorChannel)
							/ (maxColorChannel - minColorChannel));
		}

		channel->clear();

		return result;

	}
};

class KernelGenerator {

public:

	void initializeKernelUsingXavierAlgorithm(int kernelHeight,
			int kernelWeight, int channelNumber, std::vector<float> * kernel) {

		//随机数生成器初始化
		std::random_device rd;
		//使用马特赛特旋转演算法伪随机数生成器
		std::mt19937 generator(rd());

		float core = sqrt(3.0f / (kernelHeight * kernelWeight * channelNumber));

		std::uniform_real_distribution<> distribution(-core, core);

		for (int i = 0; i < kernel->size(); i++) {
			kernel->at(i) = static_cast<float>(distribution(generator));
		}

	}

	void initializeBiasUsingXavierAlgorithm(int kernelHeight, int kernelWeight,
			int channelNumber, std::vector<float> * bias) {

		//随机数生成器初始化
		std::random_device rd;
		//使用马特赛特旋转演算法伪随机数生成器
		std::mt19937 generator(rd());

		float core = sqrt(3.0f / (kernelHeight * kernelWeight * channelNumber));

		std::uniform_real_distribution<> distribution(-core, core);

		for (int i = 0; i < bias->size(); i++) {
			bias->at(i) = static_cast<float>(distribution(generator));
		}

	}

};

class Utility {

public:
	static float* VectorToArray(std::vector<float> * input) {

		float * array;
		array = new float[input->size()];

		for (int i = 0; i < input->size(); i++) {
			array[i] = input->at(i);
		}

		return array;
	}
};

class TestCase {

public:
	static void TestCase1(float * data, float * kernel, float * bias,
			float * output_data) {

		float sum = 0.0f;

		sum += data[0] * kernel[0];
		sum += data[1] * kernel[1];
		sum += data[2] * kernel[2];
		sum += data[540] * kernel[3];
		sum += data[541] * kernel[4];
		sum += data[542] * kernel[5];
		sum += data[1080] * kernel[6];
		sum += data[1081] * kernel[7];
		sum += data[1082] * kernel[8];

		std::cout << " CPU result : " << sum << std::endl;
		std::cout << " bias unit : " << bias[0] << std::endl;
		std::cout << " GPU result : " << output_data[0] << std::endl;

	}

	static void printDynamicArray(float * array, int length) {
		for (int i = 0; i < length; i++) {
			std::cout << array[i] << std::endl;
		}
	}

};

int main() {

	int filePathMaxLength = 256;
	int imageHeight = 540;
	int imageWidth = 540;
	int kernelHeight = 3;
	int kernelWidth = 3;
	//输入featuremap为1个，即图片原始数据。输出为1个featuremap，所以需要1个卷积核。
	int inputFeaturemaps = 1;
	int outputFeaturemaps = 1;

	//读取图片到RGB三个通道
	char imagePath[filePathMaxLength];
	getcwd(imagePath, filePathMaxLength);
	strcat(imagePath, "/trainingset/");
	strcat(imagePath, "train1.png");
	std::vector<float> redChannel;
	std::vector<float> greenChannel;
	std::vector<float> blueChannel;
	float * h_input_data;
	ImageProcessor processor;
	processor.readRGBImage(imagePath, &redChannel, &greenChannel, &blueChannel);
	redChannel = processor.imageChannelNormalization(&redChannel);
	greenChannel = processor.imageChannelNormalization(&greenChannel);
	blueChannel = processor.imageChannelNormalization(&blueChannel);
	h_input_data = Utility::VectorToArray(&redChannel);

	//卷积核初始化
	KernelGenerator generator;
	float * h_kernel;
	std::vector<float> kernel(kernelHeight * kernelWidth);
	generator.initializeKernelUsingXavierAlgorithm(kernelHeight, kernelWidth,
			outputFeaturemaps, &kernel);
	h_kernel = Utility::VectorToArray(&kernel);

	//偏置项初始化
	//与卷积核同样的方法，同样的数量
	float * h_bias;
	std::vector<float> bias(kernelHeight * kernelWidth);
	generator.initializeBiasUsingXavierAlgorithm(kernelHeight, kernelWidth,
			outputFeaturemaps, &bias);
	h_bias = Utility::VectorToArray(&bias);

	//GPU设定
	int GPUs;
	checkCudaErrors(cudaGetDeviceCount(&GPUs));
	if (GPUs > 0) {
		checkCudaErrors(cudaSetDevice(0));
	} else {
		return 0;
	}

	//cudnn初始化
	cudnnHandle_t cudnnHandle = NULL;
	cudnnCreate(&cudnnHandle);

	CuNeuralNetwork network;

	//输入数据设定
	cudnnTensorDescriptor_t inputDataTensor;
	float * d_data = network.createInputDataLayer(h_input_data,
			&inputDataTensor, 1, 1, imageHeight, imageWidth);

	//卷积核设定
	cudnnFilterDescriptor_t kernelDescriptor;
	float * d_kernel = network.createKernel(h_kernel, &kernelDescriptor,
			inputFeaturemaps, outputFeaturemaps, kernelHeight, kernelWidth);

	//卷积操作设定
	cudnnConvolutionDescriptor_t convolutionDescriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
	//零填充的行数与列数：0,卷积的水平和垂直的滑动长度：1,x，y向上取样的比例尺：1
	//不使用卷积操作，因为卷积操作要旋转卷积核，而这里不需要旋转，互相关就是无需旋转的卷积乘法。
	checkCUDNN(
			cudnnSetConvolution2dDescriptor(convolutionDescriptor, 0, 0, 1, 1,
					1, 1, CUDNN_CROSS_CORRELATION));

	//输出数据设定
	cudnnTensorDescriptor_t outputDataTensor;
	OutputDim outputDim;
	float *d_output_data = network.createOutputDataLayer(&inputDataTensor,
			&kernelDescriptor, &convolutionDescriptor, &outputDataTensor,
			&outputDim);

	//选择FP算法
	cudnnConvolutionFwdAlgo_t algorithm;
	checkCUDNN(
			cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDataTensor,
					kernelDescriptor, convolutionDescriptor, outputDataTensor,
					CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algorithm));

	//设置workspace的大小
	size_t workspaceSizeInByte = 0;
	checkCUDNN(
			cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
					inputDataTensor, kernelDescriptor, convolutionDescriptor,
					outputDataTensor, algorithm, &workspaceSizeInByte));
	void *d_cudnn_workspace = nullptr;
	checkCudaErrors(cudaMalloc(&d_cudnn_workspace, workspaceSizeInByte));

	checkCudaErrors(cudaDeviceSynchronize());

	//FP计算
	float alpha = 1.0f, beta = 0.0f;

	//卷积运算
	checkCUDNN(
			cudnnConvolutionForward(cudnnHandle, &alpha, inputDataTensor,
					d_data, kernelDescriptor, d_kernel, convolutionDescriptor,
					algorithm, d_cudnn_workspace, workspaceSizeInByte, &beta,
					outputDataTensor, d_output_data));

	//偏置项设定
	cudnnTensorDescriptor_t biasTensorDescriptor;
	float *d_bias = network.addBiasUnits(h_bias, &biasTensorDescriptor,
			outputFeaturemaps, kernelHeight, kernelWidth);

	//加上偏置项
	checkCUDNN(
			cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME_C, &alpha,
					biasTensorDescriptor, d_bias, &alpha, outputDataTensor,
					d_output_data));

	//输出数据回传
	float * h_output_data =
			new float[outputDim.outputImages
					* outputDim.outputFeaturemapsForEachImage
					* outputDim.outputFeaturemapHeight
					* outputDim.outputFeaturemapWidth];
	checkCudaErrors(
			cudaMemcpyAsync(h_output_data, d_output_data,
					sizeof(float) * outputDim.outputImages
							* outputDim.outputFeaturemapsForEachImage
							* outputDim.outputFeaturemapHeight
							* outputDim.outputFeaturemapWidth,
					cudaMemcpyDeviceToHost));

	//测试用例1
	TestCase::TestCase1(h_input_data, h_kernel, h_bias, h_output_data);

	//checkCUDNN(cudnnDestroyTensorDescriptor(redChannelDataTensor));

}
