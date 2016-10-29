#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "SobelFilter.h"

int main(int argc , char ** argv)
{
	cv::Mat image;
	image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR); 
	SobelFilter *instance = SobelFilter::factory();
	cv::imshow("" , image);
	cv::Mat output = (*instance)(image);
	cv::waitKey(0);
	cv::imshow( "sobel", output);
	cv::waitKey(0);
	return 0;
}
