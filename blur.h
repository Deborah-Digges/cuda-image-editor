#ifndef BLURRER_H
#define BLURRER_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>


class Blurrer
{
	private:
		cv::Mat imageInputRGBA;
		cv::Mat imageOutputRGBA;
		uchar4 * h_inputImageRGBA;
		uchar4 * h_outputImageRGBA;
		uchar4 * d_inputImageRGBA;
		uchar4 * d_outputImageRGBA;
		unsigned char *d_red;
		unsigned char *d_green;
		unsigned char *d_blue;
		float * h_filter;
		float * d_filter;
		unsigned char * d_redBlurred;
		unsigned char * d_greenBlurred;	
		unsigned char * d_blueBlurred;
		int filterWidth;
		int numRows;
		int numCols;
		static Blurrer * instance;
		
		Blurrer();
		Blurrer(const Blurrer&);
		Blurrer& operator=(const Blurrer &);		
		
		void allocateMemory();
		void makeFilter();
		void wrapperBlurrer();
		void resetBlurrer();
	
	public:
		static Blurrer * factory();
		cv::Mat operator()(const cv::Mat & image);
		~Blurrer();

};
#endif
