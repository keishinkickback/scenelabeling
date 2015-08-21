#include "TestCase.h"

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

void TestCase::TestCase1(float * data, float * kernel, float * bias, float * output_data,
		float * pooling_output_data) {

	float sum1 = data[0] * kernel[0];
	float sum2 = data[1] * kernel[1];
	float sum3 = data[2] * kernel[2];
	float sum4 = data[540] * kernel[3];
	float sum5 = data[541] * kernel[4];
	float sum6 = data[542] * kernel[5];
	float sum7 = data[1080] * kernel[6];
	float sum8 = data[1081] * kernel[7];
	float sum9 = data[1082] * kernel[8];

	std::cout << " CPU result : "
			<< sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8 + sum9
			<< std::endl;
	std::cout << " bias unit : " << bias[0] << std::endl;
	std::cout << " GPU result : " << output_data[0] << std::endl;
	std::cout << " output data value : " << output_data[0] << " "
			<< output_data[1] << " " << output_data[2] << " "
			<< output_data[538] << " " << output_data[539] << " "
			<< output_data[540] << " " << output_data[1076] << " "
			<< output_data[1077] << " " << output_data[1078] << std::endl;
	std::cout << " GPU max pooling result : " << pooling_output_data[0]
			<< std::endl;

}
