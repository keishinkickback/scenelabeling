/*
 * ImageProcessor.h
 *
 *  Created on: Aug 21, 2015
 *      Author: root
 */

#ifndef IMAGEPROCESSOR_H_
#define IMAGEPROCESSOR_H_

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <FreeImage.h>
#include <algorithm>

class ImageProcessor {

public:
	void readRGBImage(char *imagePath, std::vector<float> *redChannel,
			std::vector<float> *greenChannel, std::vector<float> *blueChannel);

	std::vector<float> imageChannelNormalization(std::vector<float> *channel);
};

#endif /* IMAGEPROCESSOR_H_ */
