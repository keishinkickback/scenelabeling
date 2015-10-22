#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <FreeImage.h>
#include <algorithm>

#include "includes/ImageProcessor.h"

void ImageProcessor::ReadRGBImage(char *imagePath,
		std::vector<float> *redChannel, std::vector<float> *greenChannel,
		std::vector<float> *blueChannel) {

	FreeImage_Initialise(TRUE);

	FIBITMAP* fib;

	FREE_IMAGE_FORMAT format = FreeImage_GetFileType(imagePath);
	fib = FreeImage_Load(format, imagePath);
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
