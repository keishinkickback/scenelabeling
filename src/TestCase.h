/*
 * TestCase.h
 *
 *  Created on: Aug 21, 2015
 *      Author: root
 */

#ifndef TESTCASE_H_
#define TESTCASE_H_

class TestCase {

public:

	static void TestbedOfConvolutionMethodForOneOutputFeaturemap(
			float * inputData, float * gpuData, float * kernel, float bias,
			int inputFeatureHeight, int inputFeaturemapWidth, int kernelHeight,
			int kernelWidth);

	static void TestbedOfMaxPoolingMethodForOneOutputFeaturemap(
			float * inputData, float * gpuData, int inputFeaturemapHeight,
			int inputFeaturemapWidth, int windowHeight, int windowWidth);

};

#endif /* TESTCASE_H_ */
