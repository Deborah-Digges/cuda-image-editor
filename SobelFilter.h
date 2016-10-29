#ifndef SOBEL_H
#define SOBEL_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

class SobelFilter
{
	private:
		cv::Mat imageInputRGBA;
		cv::Mat imageOutputRGBA;
		uchar4 * h_inputImageRGBA;
		uchar4 * h_outputImageRGBA;
		uchar4 * d_inputImageRGBA;
		uchar4 * d_outputImageRGBA;
		float * h_filterX;
		float * d_filterX;
		float * h_filterY;
		float * d_filterY;
		int filterWidth;
		int numRows;
		int numCols;
		static SobelFilter * instance;
		
		SobelFilter();
		SobelFilter(const SobelFilter&);
		SobelFilter& operator=(const SobelFilter&);		
		
		void allocateMemory();
		void makeFilter();
		void wrapperFilter();
		void resetFilter();
	
	public:
		static SobelFilter * factory();
		cv::Mat operator()(const cv::Mat & image);
		~SobelFilter();

};
#endif
