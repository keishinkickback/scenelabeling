#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <FreeImage.h>
#include <algorithm>

#include "ImageProcessor.h"

void ImageProcessor::readRGBImage(char *imagePath,
		std::vector<float> *redChannel, std::vector<float> *greenChannel,
		std::vector<float> *blueChannel) {

	FreeImage_Initialise (TRUE);

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

std::vector<float> ImageProcessor::imageChannelNormalization(
		std::vector<float> *channel) {

	float maxColorChannel = *std::max_element(channel->begin(), channel->end());
	float minColorChannel = *std::min_element(channel->begin(), channel->end());

	std::vector<float> result;

	for (int i = 0; i < channel->size(); i++) {
		result.push_back(
				(channel->at(i) - minColorChannel)
						/ (maxColorChannel - minColorChannel));
	}

	channel->clear();

	return result;

}
