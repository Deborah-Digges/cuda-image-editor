#ifndef BW_H
#define BW_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

class BWConverter
{	
	public:
		cv::Mat operator()(const cv::Mat& image);
		static BWConverter * factory(); 
		~BWConverter();
		
	
	private:
		cv::Mat imageRGBA;
		cv::Mat imageGrey;
		uchar4 *h_rgbaImage;
		uchar4 *d_rgbaImage;
		unsigned char *h_greyImage;
		unsigned char *d_greyImage;
		size_t numRows;
		size_t numCols;
		size_t step;
		static BWConverter * instance;
		
		//singleton : private constructor , copy constructor and assignment operator
		BWConverter();
		BWConverter& operator=(const BWConverter&);
		BWConverter(const BWConverter&);
		
		
		void preProcess();	
		void wrapper_rgba_to_greyscale();
		void resetConverter();
	
		
		

};

#endif
