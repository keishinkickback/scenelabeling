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
#include <math.h>

float* Utility::VectorToArray(std::vector<float> * input) {

	float * array;
	array = new float[input->size()];

	for (int i = 0; i < input->size(); i++) {
		array[i] = input->at(i);
	}

	return array;
}

void Utility::PrintDynamicArray(float * array, int length) {
	for (int i = 0; i < length; i++) {
		std::cout << array[i] << std::endl;
	}
}

std::vector<std::vector<float> > Utility::ArrayToMatrix(float * array,
		int height, int width) {

	int arrayIndex = 0;
	std::vector<std::vector<float> > matrix;

	for (int y = 0; y < height; y++) {

		std::vector<float> row;

		for (int x = 0; x < width; x++) {

			row.push_back(array[arrayIndex]);
			arrayIndex++;

		}

		matrix.push_back(row);

	}

	return matrix;

}

bool Utility::floatIsEqual(float a, float b) {

	if (abs(a - b) <= 0.00001f) {

		return true;

	} else {

		return false;

	}

}
