#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include "blur.h"

/* Declaration of kernel functions.
 */
__global__ void separateChannels(const uchar4* const inputImageRGBA , int numRows , int numCols , unsigned char* const redChannel , 
                      unsigned char* const greenChannel , unsigned char* const blueChannel);

__global__
void recombineChannels(const unsigned char* const redChannel , const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols);

__global__
void gaussianBlur(const unsigned char* const inputChannel , unsigned char* const outputChannel, int numRows , int numCols , const float* const filter, int filterWidth);


Blurrer * Blurrer::instance = NULL;

/* Private constructor for singleton class */
Blurrer::Blurrer()
{
}

/* The client can create an object of Blurrer class only through the static factory function.
 * It returns a pointer to the only currently existing instance of the class.
 */
Blurrer * Blurrer::factory()
{
	if(Blurrer::instance == NULL)
	{
		Blurrer::instance = new Blurrer();
	}
	return Blurrer::instance;
	
}

/* The input image is converted to the appropriate format for processing.
 * Helper methods are called for allocating memory on the GPU and copying from the CPU to GPU , making the filter and performing the blurring
 * The output image is copied from GPU to CPU memory
 * The output image is converted to cv::Mat format and returned to the client.
 */

cv::Mat Blurrer::operator()(const cv::Mat &image)
{
	cv::cvtColor(image , imageInputRGBA , CV_BGR2RGBA);
	numRows = imageInputRGBA.rows;
	numCols = imageInputRGBA.cols;
	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);
	allocateMemory();
	makeFilter();
	wrapperBlurrer();
	cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA , sizeof(uchar4) * numRows * numCols, cudaMemcpyDeviceToHost);
	cv::Mat output(imageOutputRGBA.rows, imageOutputRGBA.cols, CV_8UC4, (void*)h_outputImageRGBA);
	resetBlurrer();
	return output;

}

/* Allocates memory on the GPU for the structures needed and initializes them to 0.
 * Copies the source image from the CPU to the GPU memory.
 */

void Blurrer::allocateMemory()
{
	const size_t numPixels = numRows * numCols;
	h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
	h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);
	cudaMalloc(&d_inputImageRGBA, sizeof(uchar4) * numPixels);
	cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numPixels);
	cudaMemset(d_outputImageRGBA, 0, numPixels * sizeof(uchar4));
	cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
	cudaMalloc(&d_redBlurred,    sizeof(unsigned char) * numPixels);
	cudaMalloc(&d_greenBlurred,  sizeof(unsigned char) * numPixels);
	cudaMalloc(&d_blueBlurred,   sizeof(unsigned char) * numPixels);
	cudaMemset(d_redBlurred,   0, sizeof(unsigned char) * numPixels);
	cudaMemset(d_greenBlurred, 0, sizeof(unsigned char) * numPixels);
	cudaMemset(d_blueBlurred,  0, sizeof(unsigned char) * numPixels);
	cudaMalloc(&d_red,   sizeof(unsigned char) * numRows * numCols);
	cudaMalloc(&d_green, sizeof(unsigned char) * numRows * numCols);
	cudaMalloc(&d_blue,  sizeof(unsigned char) * numRows * numCols);
	

}

/* Creates a 9x9 filter which will be used for convolution with the input image to create the blurred image.
 * The blure kernel is created in the host memory,
 * It is then copied into the GPU memory.
 */

void Blurrer::makeFilter()
{
	const int blurKernelWidth = 9;
	const float blurKernelSigma = 2.;
	filterWidth = blurKernelWidth;

	//create and fill the filter we will convolve with
	h_filter = new float[blurKernelWidth * blurKernelWidth];
	float filterSum = 0.f; //for normalization
	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) 
	{
		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) 
		{
			float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      			h_filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      			filterSum += filterValue;
    		}	
  	}

  	float normalizationFactor = 1.f / filterSum;

  	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) 
	{
    		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) 
		{
			h_filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    		}
  	}	
	
	cudaMalloc(&d_filter , filterWidth * filterWidth * sizeof(float));
	cudaMemcpy(d_filter , h_filter , filterWidth * filterWidth * sizeof(float) , cudaMemcpyHostToDevice);
}

/* Wrapper to call the blur kernel.
 * Initializes the block and grid dimensions and calls a series of 3 kernels  : 
 * seperateChannels , gaussianBlur : once for each colour channel.
 * recombineChannels
 */
void Blurrer::wrapperBlurrer()
{
	const int BLOCK_WIDTH =  32;
	const dim3 blockSize(BLOCK_WIDTH , BLOCK_WIDTH);
	const dim3 gridSize((numCols/BLOCK_WIDTH) + 1 , (numRows/BLOCK_WIDTH) + 1 );
	
	separateChannels<<<gridSize,blockSize>>>(d_inputImageRGBA , numRows , numCols, d_red, d_green , d_blue);
        cudaDeviceSynchronize(); cudaGetLastError();
  
	gaussianBlur<<<gridSize , blockSize>>>(d_red , d_redBlurred, numRows, numCols, d_filter, filterWidth);
	gaussianBlur<<<gridSize , blockSize>>>(d_green , d_greenBlurred, numRows, numCols, d_filter, filterWidth);
	gaussianBlur<<<gridSize , blockSize>>>(d_blue , d_blueBlurred, numRows, numCols, d_filter, filterWidth);
	cudaDeviceSynchronize(); cudaGetLastError();
  
	recombineChannels<<<gridSize, blockSize>>>(d_redBlurred , d_greenBlurred , d_blueBlurred , d_outputImageRGBA , numRows, numCols);
	cudaDeviceSynchronize(); cudaGetLastError();
}

/* After each image has been processed, the blurrer is reset so that the data structures for each image are created according to the 
 * size of the input image.
 * All instance members are taken care of here.
 * The destructor here is therefore a dummy destructor.
 */
void Blurrer::resetBlurrer()
{
	cudaFree(d_red);
	cudaFree(d_green);
	cudaFree(d_blue);
	cudaFree(d_filter);
	cudaFree(d_redBlurred);	
	cudaFree(d_greenBlurred);
	cudaFree(d_blueBlurred);
	cudaFree(d_inputImageRGBA);
	cudaFree(d_outputImageRGBA);
	delete [] h_filter;
	h_inputImageRGBA = NULL;
	h_outputImageRGBA = NULL;
	d_inputImageRGBA = NULL;
	d_outputImageRGBA = NULL;
	d_red = NULL;
	d_green = NULL;
	d_blue = NULL;
	h_filter = NULL;
	d_filter = NULL;
	d_redBlurred = NULL;
	d_greenBlurred = NULL; 	
	d_blueBlurred = NULL;
	filterWidth = 0;
	numRows = 0;
	numCols = 0;
}


Blurrer::~Blurrer()
{
	
}

/* Performs the actual blurring , by convolution.
 * The center of the filter is aligned with each pixel , for every pixel in the channel.
 * The corresponding values are multiplied and their sum is found.
 * This is the value placed in the corresponding pixel in the output image.
 */
__global__
void gaussianBlur(const unsigned char* const inputChannel , unsigned char* const outputChannel, int numRows , int numCols , const float* const filter, int filterWidth)
{
	int half_width = filterWidth/2;
	float image_value = 0.0f;
	float blur_value = 0.0f;
	float computed_value = 0.0f;
	int row = 0;
	int column = 0;
	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        {
		return;
	}

	for(row = -half_width ; row <= half_width ; ++row)
	{
		for(column = -half_width; column <= half_width ; ++column)
		{
			int image_r = min(max(thread_2D_pos.y + row, 0), (numRows - 1));
			int image_c = min(max(thread_2D_pos.x + column, 0), (numCols - 1));
			image_value = static_cast<float>(inputChannel[(image_r) * numCols + (image_c)]);
			blur_value = filter[(row + half_width) * filterWidth + (column + half_width)];
			computed_value += image_value * blur_value;
                           
            
		}
       
	}
    
	outputChannel[thread_1D_pos] = static_cast<char>(computed_value);
	
}



/* This kernel takes in an image where each pixel is represented as a uchar4 and splits
 * it into three color channels . 
 */
__global__ void separateChannels(const uchar4* const inputImageRGBA , int numRows , int numCols , unsigned char* const redChannel , 
                      unsigned char* const greenChannel , unsigned char* const blueChannel)
{
	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)	
	{
		return;
	}

	redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
	greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
	blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;

}

/* Recombines the three colour channels to form a single output coloured image */
__global__
void recombineChannels(const unsigned char* const redChannel , const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
	{
		return;
	}

	unsigned char red   = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue  = blueChannel[thread_1D_pos];
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);
	outputImageRGBA[thread_1D_pos] = outputPixel;
}






