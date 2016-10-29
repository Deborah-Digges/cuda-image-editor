#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include "SobelFilter.h"
#include <cmath>

/* Declaration of kernel functions.
 */
__global__ void sobelFilter(const uchar4* d_inputImageRGBA , uchar4* d_outputImageRGBA , const float* d_filterX , const float* d_filterY , int numRows , int numCols , int filterWidth);
__device__ unsigned char gray(const uchar4 & pix);
__device__ char threshold(char input);


SobelFilter * SobelFilter::instance = NULL;


SobelFilter::SobelFilter()
{
}

/* The client can create an object of SobelFilter class only through the static factory function.
 * It returns a pointer to the only currently existing instance of the class.
 */
SobelFilter * SobelFilter::factory()
{
	if(SobelFilter::instance == NULL)
	{
		SobelFilter::instance = new SobelFilter();
	}
	return SobelFilter::instance;
	
}

/* The input image is converted to the appropriate format for processing.
 * Helper methods are called for allocating memory on the GPU and copying from the CPU to GPU , making the filter and performing the blurring
 * The output image is copied from GPU to CPU memory
 * The output image is converted to cv::Mat format and returned to the client.
 */
cv::Mat SobelFilter::operator()(const cv::Mat & image)
{
	std::cout << "functor\n";
	cv::cvtColor(image , imageInputRGBA , CV_BGR2RGBA);
	numRows = imageInputRGBA.rows;
	numCols = imageInputRGBA.cols;
	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);
	allocateMemory();
	makeFilter();
	wrapperFilter();
	cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA , sizeof(uchar4) * numRows * numCols, cudaMemcpyDeviceToHost);
	cv::Mat output(imageOutputRGBA.rows, imageOutputRGBA.cols, CV_8UC4, (void*)h_outputImageRGBA);
	resetFilter();							//CV_8UC4
	return output;

}

/* Allocates memory on the GPU for the structures needed and initializes them to 0.
 * Copies the source image from the CPU to the GPU memory.
 */

void SobelFilter::allocateMemory()
{
	std::cout << "allocate\n";
	const size_t numPixels = numRows * numCols;
	h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
	h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);
	cudaMalloc(&d_inputImageRGBA, sizeof(uchar4) * numPixels);
	cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numPixels);
	cudaMemset(d_outputImageRGBA, 0, numPixels * sizeof(uchar4));
	cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
	
}
/* Creates two 3x3 filters for differentiation of the image intensity function in both the x and the y direction.
 * The 2 filters are created in host memory and then copied to GPU memory.
 */


void SobelFilter::makeFilter()
{
	std::cout << "make\n";
	filterWidth = 3;
	int index = 0;
	int x[] = {-1 , 0 , 1 , -2 , 0 , 2 , -1 , 0 , 1};
	int y[] = {1 , 2 , 1 , 0 , 0 , 0 , -1 , -2 , -1};
	//int y[] = {3 , 10 , 3 , 0 , 0 , 0 , -3 , -10 , -3};
	//int x[] = {3 , 0  , -3 , 10 , 0 , -10 , 3 , 0 , -3};
	//create and fill the filter we will convolve with
	h_filterX = new float[filterWidth * filterWidth];
	h_filterY = new float[filterWidth * filterWidth];
	
	for(index = 0; index < filterWidth * filterWidth; ++index)
	{
		h_filterX[index] = x[index];
		h_filterY[index] = y[index];	
	}	
		
	
	cudaMalloc(&d_filterX , filterWidth * filterWidth * sizeof(float));
	cudaMalloc(&d_filterY , filterWidth * filterWidth * sizeof(float));
	cudaMemcpy(d_filterX , h_filterX , filterWidth * filterWidth * sizeof(float) , cudaMemcpyHostToDevice);
	cudaMemcpy(d_filterY , h_filterY , filterWidth * filterWidth * sizeof(float) , cudaMemcpyHostToDevice);
}

/* Wrapper to call the sobel kernel.
 * Initializes the block and grid dimensions and calls the sobel filter kernel 
*/
void SobelFilter::wrapperFilter()
{
	std::cout << "reset\n";
	const int BLOCK_WIDTH =  32;
	const dim3 blockSize(BLOCK_WIDTH , BLOCK_WIDTH);
	const dim3 gridSize((numCols/BLOCK_WIDTH) + 1 , (numRows/BLOCK_WIDTH) + 1 );
	
	sobelFilter<<<gridSize,blockSize>>>(d_inputImageRGBA , d_outputImageRGBA , d_filterX , d_filterY , numRows , numCols , filterWidth);
        cudaDeviceSynchronize(); cudaGetLastError();
  	
}

/* Helper function for returning a grey scale pixel for a given colored pixel */
__device__ unsigned char gray(const uchar4 & pix)
{
	return (0.3f * pix.x + 0.59f * pix.y + 0.11f * pix.z);
}

/* Kernel that performs the edge detection using sobel filter.
 * It first applies by convolution the filter for differentiation in the x direction and then the filter for diff. in the y direction.
 * The magnitude of the differential of the image intensity function is found by pythogoras theorem.
 * The magnitude is compared to a threshold value; If it is greater than a threshold value, the pixel is made black , else the pixel is made 
 * white.
 */
__global__ void sobelFilter(const uchar4* d_inputImageRGBA , uchar4* d_outputImageRGBA , const float* d_filterX , const float* d_filterY , int numRows , int numCols , int filterWidth)
{
	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x , blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        {
		return;
	}
		
	float sumX = 0.f;
	float sumY = 0.f;
	int row = 0;
	int col = 0;
	float g = 0.f;
	float filterX_value =  0.f;
	float filterY_value = 0.f;
	int halfWidth = filterWidth/2;
	
	for(row = -1 ; row <= 1 ; ++row)
	{
		for(col = -1 ; col <= 1 ; ++col)
		{
			int image_r = min(max(thread_2D_pos.y + row, 0), (numRows - 1));
			int image_c = min(max(thread_2D_pos.x + col, 0), (numCols - 1));
			g = gray(d_inputImageRGBA[(image_r) * numCols + (image_c)]);
			filterX_value = d_filterX[(row + halfWidth) * filterWidth + (col + halfWidth)];
			filterY_value = d_filterY[(row + halfWidth) * filterWidth + (col + halfWidth)];
			sumX += g * filterX_value;
			sumY += g * filterY_value;
                        
		}
	}
	//unsigned char p = abs((sumX/8.0)) + abs((sumY/8.0));
	unsigned char p = sqrt( abs((sumX/8.0)) * abs((sumX/8.0)) + abs((sumY/8.0)) * abs((sumY/8.0)));
	p = threshold(p);
	uchar4 output;
	output.x = 0;
	output.y = 0;
	output.z = 0;
	output.w = p;
	
	d_outputImageRGBA[thread_1D_pos] = output;//make_uchar4(0, 255 , 255 , 255);	
	
	
}

__device__ char threshold(char input)
{
	if ( input >= 8 )
	{
		return (char )255;
	}
	else
	{
		return (char )0 ;
	}
} 

/* After each image has been processed, the SobelFilter object is reset so that the data structures for each image are created according 
 * the size of the input image.
 * All instance members are taken care of here.
 * The destructor here is therefore a dummy destructor.
 */
void SobelFilter::resetFilter()
{
	std::cout <<"reset\n";
	cudaFree(d_inputImageRGBA);
	cudaFree(d_outputImageRGBA);
	cudaFree(d_filterX);
	cudaFree(d_filterY);
	delete [] h_filterX;
	delete [] h_filterY;
	h_inputImageRGBA = NULL;
	h_outputImageRGBA = NULL;
	d_inputImageRGBA = NULL;
	d_outputImageRGBA = NULL;
	h_filterX = NULL;
	h_filterY = NULL;
	d_filterX = NULL;
	d_filterY = NULL;
	filterWidth = 0;
	numRows = 0;
	numCols = 0;
}




