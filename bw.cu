#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "bw.h"

BWConverter::BWConverter()
{
}

/* Declaration of kernel functions.*/
BWConverter * BWConverter::instance = NULL;
__global__ void rgba_to_greyscale(const uchar4* rgbaImage , unsigned char* greyImage , int numRows , int numCols);


/* After each image has been processed, the converter is reset so that the data structures for each image are created according to the 
 * size of the input image.
 * All instance members are taken care of here.
 * The destructor here is therefore a dummy destructor.
 */
void BWConverter::resetConverter()
{
	cudaFree(d_rgbaImage);
	cudaFree(d_greyImage);
	h_rgbaImage = NULL;
	d_rgbaImage = NULL;
	h_greyImage = NULL;
	d_greyImage = NULL;
	numRows = 0;
	numCols = 0;
	step = 0;
		
}

/* The input image is converted to the appropriate format for processing.
 * Helper methods are called for allocating memory on the GPU and copying from the CPU to GPU , making the filter and performing the blurring
 * The output image is copied from GPU to CPU memory
 * The output image is converted to cv::Mat format and returned to the client.
 */
cv::Mat BWConverter::operator()(const cv::Mat& image)
{
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);
	numRows = imageRGBA.rows;
	numCols = imageRGBA.cols;	
	step = imageRGBA.step;
	imageGrey.create(image.rows, image.cols, CV_8UC1 );
	preProcess();
	wrapper_rgba_to_greyscale();
	cudaDeviceSynchronize();
	cudaGetLastError();
	cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage, sizeof(unsigned char) * numRows * numCols, cudaMemcpyDeviceToHost);
	resetConverter();
	return imageGrey;
	
}

/* Allocates memory on the GPU for the structures needed and initializes them to 0.
 * Copies the source image from the CPU to the GPU memory.
 */
void BWConverter::preProcess()
{
	h_rgbaImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	h_greyImage  = (unsigned char *)imageGrey.ptr<unsigned char>(0);
	cudaMalloc(&d_rgbaImage, sizeof(uchar4) * numRows * numCols);
	cudaMalloc(&d_greyImage, sizeof(unsigned char) * numRows * numCols);
	cudaMemset(d_greyImage, 0, numRows * numCols * sizeof(unsigned char)); //make sure no memory is left laying around
	cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * numRows * numCols, cudaMemcpyHostToDevice);
}

/* Kernel that takes an image in RGBA and converts it to black and white, by performing the following function on each pixel(map operation)
 * output_pixel = .299f * (rgb.x) + .587f * (rgb.y) + .114f * (rgb.z);
 */ 
__global__ void rgba_to_greyscale(const uchar4* rgbaImage , unsigned char* greyImage , int numRows , int numCols)
{
	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x , blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
	{
		return;
	}
    
	uchar4 rgb = rgbaImage[thread_1D_pos];
	greyImage[thread_1D_pos] = .299f * (rgb.x) + .587f * (rgb.y) + .114f * (rgb.z);
}

/* Wrapper to call the kernel.
 * Initializes the block and grid dimensions and calls the black and white kernel.
 */
void BWConverter::wrapper_rgba_to_greyscale()
{
	const int BLOCK_SIZE = 32;
	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); 
	const dim3 gridSize((numCols/BLOCK_SIZE) + 1, (numRows/BLOCK_SIZE) + 1);  
	rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  	cudaDeviceSynchronize();
	cudaGetLastError();

}

/* Dummy destructor */
BWConverter::~BWConverter()
{


}

/* The client can create an object of BWConverter class only through the static factory function.
 * It returns a pointer to the only currently existing instance of the class.
 */

BWConverter * BWConverter::factory()
{
	if(BWConverter::instance == NULL)
	{
		BWConverter::instance = new BWConverter();
	}
	return BWConverter::instance;
	
}


