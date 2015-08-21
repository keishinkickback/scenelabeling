#include "Utility.h"

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

float* Utility::VectorToArray(std::vector<float> * input) {

	float * array;
	array = new float[input->size()];

	for (int i = 0; i < input->size(); i++) {
		array[i] = input->at(i);
	}

	return array;
}

void Utility::printDynamicArray(float * array, int length) {
	for (int i = 0; i < length; i++) {
		std::cout << array[i] << std::endl;
	}
}
