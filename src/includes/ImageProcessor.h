#ifndef IMAGEPROCESSOR_H_
#define IMAGEPROCESSOR_H_

#include <FreeImage.h>
#include <algorithm>

class ImageProcessor {

public:

	//读取具有RGB三通道的图片
	static void ReadRGBImage(char *imagePath, std::vector<float> *redChannel,
			std::vector<float> *greenChannel, std::vector<float> *blueChannel);

	//正规化图片通道数据
	static std::vector<float> Normalization(std::vector<float> *channel);
};

#endif /* IMAGEPROCESSOR_H_ */
