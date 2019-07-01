#include <iostream>
#include <vector>//Thread building blocks library
#include <tbb/task_scheduler_init.h>//Free Image library
#include <FreeImagePlus.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_reduce.h>
#include <mutex>
#include <math.h>
#include <tbb/tick_count.h>
#include <chrono>
#include <fstream>
std::mutex ilock;

using namespace std; 
using namespace tbb; 
using namespace std::chrono;

const int KERNEL_SIZE = 25; 
float kernelSize[KERNEL_SIZE][KERNEL_SIZE] = { 0 };
const int PI = 3.142f; 
template <typename T>T sqr(const T& a) {
	return a * a; 
}
double sum = 0; 
float sigma = 10.5; 
float gaussian2D(float x, float y, float sigma) 
{ 
	return 1.0f / (2.0f*PI*sqr(sigma)) * exp(-((sqr(x) + sqr(y)) / (2.0f*sqr(sigma)))); 
}
int kernelCenter = KERNEL_SIZE / 2; 
float kernel[KERNEL_SIZE][KERNEL_SIZE];

void parallelBlur(int stepSize) {
	fipImage nickysImage; 
	nickysImage.load("../Images/render_1.png"); 
	nickysImage.convertToFloat(); 
	unsigned int width = nickysImage.getWidth(); 
	unsigned int height = nickysImage.getHeight();
	const float* const inputBuffer = (float*)nickysImage.accessPixels(); 
	ofstream file; 
	fipImage outputNickysImage; 
	outputNickysImage = fipImage(FIT_FLOAT, width, height, 32); 
	float *floatOutput = (float*)outputNickysImage.accessPixels(); 
	tick_count t0 = tick_count::now();
	for (int x = -kernelCenter; x < kernelCenter; x++) 
	{ 
		for (int y = -kernelCenter; y < kernelCenter; y++) 
		{
			kernel[x + kernelCenter][y + kernelCenter] = 1.0f / (2.0f*PI*sqr(sigma)) * exp(-((sqr(x) + sqr(y)) / (2.0f*sqr(sigma)))); 
			sum += kernel[x][y]; 
		} 
	}
	for (int i = 0; i < KERNEL_SIZE; i++) 
	{ 
		for (int j = 0; j < KERNEL_SIZE; j++) { kernel[i][j] /= sum; 
		} 
	}

	parallel_for(blocked_range2d<int, int>(kernelCenter, height - kernelCenter, stepSize, kernelCenter, width - kernelCenter, stepSize), [&](const blocked_range2d<int, int>& r) {
		auto y1 = r.rows().begin(); 
		auto y2 = r.rows().end(); 
		auto x1 = r.cols().begin(); 
		auto x2 = r.cols().end(); 
		for (int y = y1; y < y2; y++) {
			for (int x = x1; x < x2; x++) {
				for (int i = 0; i < KERNEL_SIZE; i++) {
					for (int j = 0; j < KERNEL_SIZE; j++) {
						floatOutput[y * width + x] += inputBuffer[(y - (i - kernelCenter)) * width + (x - (j - kernelCenter))] * kernel[i][j];//gaussian2D(i,j,sigma); //kernelSize[i][j];
					}
				}
			}
		}
	});

	outputNickysImage.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP); 
	outputNickysImage.convertTo24Bits(); 
	outputNickysImage.save("RGB_processedBlurImage2.png"); 
	tick_count t1 = tick_count::now();
	file.open("../Images/testing.csv", fstream::app);
	//file << "Kernel Size" << "," << KERNEL_SIZE << endl;
	//file << "StepSize" << "," << stepSize << endl;
	file << "Time: " << "," << (t1 -t0).seconds() << endl;
	file.close();
	cout << "Question 1: using dynamic kernel " << KERNEL_SIZE <<" with Parallel for took: " << (t1 -t0).seconds() << " To complete with stepSize " << stepSize << endl;
}

void sequentialBlur() {
	fipImage nickysImage; 
	nickysImage.load("../Images/render_1.png"); 
	nickysImage.convertToFloat(); 
	unsigned int width = nickysImage.getWidth(); 
	unsigned int height = nickysImage.getHeight(); 
	const float* const inputBuffer = (float*)nickysImage.accessPixels(); 
	ofstream file2; 
	fipImage outputNickysImage;
	outputNickysImage = fipImage(FIT_FLOAT, width, height, 32); 
	float *floatOutput = (float*)outputNickysImage.accessPixels(); 
	tick_count t0 = tick_count::now();

	for (int x = -kernelCenter; x < kernelCenter; x++)
	{ 
		for (int y = -kernelCenter; y < kernelCenter; y++) 
		{ 
			kernel[x + kernelCenter][y + kernelCenter] = 1.0f / (2.0f*PI*sqr(sigma)) * exp(-((sqr(x) + sqr(y)) / (2.0f*sqr(sigma))));
			sum += kernel[x][y]; 
		} 
	}

	for (int i = 0; i < KERNEL_SIZE; i++) 
	{ 
		for (int j = 0; j < KERNEL_SIZE; j++) 
		{ 
			kernel[i][j] /= sum; 
		} 
	}

	int kernelCenter = KERNEL_SIZE / 2; 
	for (int y = kernelCenter; y < height - kernelCenter; y++) {
		for (int x = kernelCenter; x < width - kernelCenter; x++) {
			for (int i = 0; i < KERNEL_SIZE; i++) {
				for (int j = 0; j < KERNEL_SIZE; j++) {
					floatOutput[y * width + x] += inputBuffer[(y - (i - kernelCenter)) * width + (x - (j - kernelCenter))] * kernel[i][j];
				}
			}
		}
	}

	outputNickysImage.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP); 
	outputNickysImage.convertTo24Bits(); 
	outputNickysImage.save("RGB_processedBlurImage2.png"); 
	tick_count t1 = tick_count::now(); 
	file2.open("../Images/testing.csv", fstream::app);
	//file << "Kernel Size" << "," << KERNEL_SIZE << endl;
	//file << "StepSize" << "," << stepSize << endl;
	file2 << "Time: " << "," << (t1 -t0).seconds() << endl;file2.close();
	cout << "Question 1: using dynamic kernel " << KERNEL_SIZE << " with Sequential for took: " << (t1 -t0).seconds() << endl;
}

int main()
{
    int nt = task_scheduler_init::default_num_threads();
    task_scheduler_init T(nt);
	int stepSize;
    //Part 1 (Greyscale Gaussian blur): -----------DO NOT REMOVE THIS COMMENT----------------------------//
	for (int u = 0; u < 10; u++)
	{
		sequentialBlur();
	}
	//for (int u = 0; u < 10; u++)
	//{
	//	parallelBlur(8); //8
	//}
	//for (int u = 0; u < 10; u++)
	//{
	//	parallelBlur(64); // 64
	//}
	//for (int u = 0; u < 10; u++)
	//{
	//	parallelBlur(256); // 256
	//}
	//for (int u = 0; u < 10; u++)
	//{
	//	parallelBlur(1000); // 1k
	//}
	//for (int u = 0; u < 10; u++)
	//{
	//	parallelBlur(4000); //4k
	//}


	//system("pause");
    //Part 2 (Colour image processing): -----------DO NOT REMOVE THIS COMMENT----------------------------//

    // Setup Input image array
	fipImage inputImage, inputImage2;
    inputImage.load("../Images/render_1.png");
	inputImage2.load("../Images/render_2.png");

    unsigned int width = inputImage.getWidth();
    unsigned int height = inputImage.getHeight();

    // Setup Output image array
    fipImage outputImage;
    outputImage = fipImage(FIT_BITMAP, width, height, 24);

    //2D Vector to hold the RGB colour data of an image
    vector<vector<RGBQUAD>> rgbValues;
    rgbValues.resize(height, vector<RGBQUAD>(width));
	int whiteCount = 0;



	parallel_for(
		blocked_range2d<int>(0, height, 0, width), 
		[&](const blocked_range2d<int>&r) {
		for (int x = r.cols().begin(); x != r.cols().end(); x++) {
			for (int y = r.rows().begin(); y != r.rows().end(); y++) {
				RGBQUAD rgb;  //FreeImage structure to hold RGB values of a single pixel
				RGBQUAD rbg2;inputImage.getPixelColor(x, y, &rgb); //Extract pixel(x,y) colour data and place it in rgb
				inputImage2.getPixelColor(x, y, &rbg2); //Extract pixel(x,y) colour data and place it in rgb

				rgbValues[y][x].rgbRed = abs(rgb.rgbRed -rbg2.rgbRed);
				rgbValues[y][x].rgbGreen = abs(rgb.rgbGreen -rbg2.rgbGreen);
				rgbValues[y][x].rgbBlue = abs(rgb.rgbBlue -rbg2.rgbBlue);

				if (rgbValues[y][x].rgbRed > 5 || rgbValues[y][x].rgbGreen > 5 || rgbValues[y][x].rgbBlue > 5)
				{
					rgbValues[y][x].rgbRed = 255;
					rgbValues[y][x].rgbGreen = 255;
					rgbValues[y][x].rgbBlue = 255;
				}
			}
		}
	});

	//int numPixels;
	//Place the pixel colour values into output image
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			outputImage.setPixelColor(x, y, &rgbValues[y][x]);
		}
	}

	int x = parallel_reduce(
		blocked_range2d<int, int>(0, height, 0, width), 0, [&](const blocked_range2d<int, int>&range, int whiteCount)->int {
		auto y1 = range.rows().begin(); 
		auto y2 = range.rows().end(); 
		auto x1 = range.cols().begin(); 
		auto x2 = range.cols().end();

		RGBQUAD rbgQuad; 
		for (auto y = y1; y < y2; ++y) 
		{ 
			for (auto x = x1; x < x2; ++x) 
			{ 
				outputImage.getPixelColor(x, y, &rbgQuad); 
				if (rbgQuad.rgbRed == 255 && rbgQuad.rgbGreen == 255 && rbgQuad.rgbBlue == 255) 
				{ 
					whiteCount++; 
				} 
			} 
		}
		return whiteCount; 
	}, 
		[&](int x, int y)->int 
	{
		return x + y; 
	});

	cout << "Number of white pixels in the image is: " << x << endl;
	float numPixels = outputImage.getWidth() * outputImage.getHeight(); 
	float percentage = (x / numPixels) * 100; cout << "White pixel Percentage = " << percentage << "%" << endl;
	//Save the processed image
	outputImage.save("RGB_processed2.png"); 
	system("pause"); 
	return 0;
}