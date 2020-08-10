#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "kernel.h"
#include <stdio.h>

using namespace cv;

#define INIT_SIZE 16*16
#define INIT_LINE 16

void putimage(Mat img, BYTE *image_dst, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = image_dst[3 * i*width + 3 * j + 0];
			img.at<Vec3b>(i, j)[1] = image_dst[3 * i*width + 3 * j + 1];
			img.at<Vec3b>(i, j)[2] = image_dst[3 * i*width + 3 * j + 2];
		}
	}
}

int k_means_picture(const char* img)
{
	Mat image;
	image = imread(img);
	Mat dst;
	cudaError_t cudaStatus;
	int width = image.cols;
		//(int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = image.rows;
		//(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int block_size = INIT_LINE;
	int block_width = width % block_size ? width / block_size + 1 : width / block_size;
	int block_height = height % block_size ? height / block_size + 1 : height / block_size;
	dim3 Info(width, height, block_size);
	int size = width * height;
	// 최대 블럭당 mean 개수
	int max_means = (int)sqrt(size / 2);
	// 현재 블럭당 mean 개수
	int cur_means = 5;
	//(int)sqrt(INIT_SIZE / 2);
	BYTE *mean;
	BYTE *label;
	// means의 전체 크기 -> 가장 많은 초기값으로 초기화
	int mean_size = (int)cur_means * block_width * block_height;
	// 
//	dim3 data_vec(block_size, block_size, cur_means);
	// cuda 사용할때 넣을 변수.
	// 한 쓰레드당 하나의 블럭을 수행하며, 이에 대한 블럭의 크기를 세팅해준다.
	dim3 thread(block_width, block_height);
	BYTE* d_mean;
	BYTE* d_img;
	BYTE* d_dstimg;
	BYTE* d_label;
	int* d_counter;
	BYTE* image_debug = (BYTE*)calloc(sizeof(BYTE), size * 3);

	BYTE* image_dst = (BYTE*)calloc(sizeof(BYTE), size * 3);
	int *mean_counter = (int*)calloc(sizeof(int), mean_size);
	// cur_means: init means number, height / INIT_LINE + 1: block height of max case, width / INIT_LINE + 1: block width of max case
	mean = (BYTE*)calloc(sizeof(BYTE *), 3 * mean_size);
	label = (BYTE*)calloc(sizeof(BYTE *), size);

	cudaStatus = cudaMalloc((void**)&d_mean, 3 * mean_size * sizeof(BYTE));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&d_counter, mean_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&d_label, size * sizeof(BYTE));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&d_img, 3 * size * sizeof(BYTE));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&d_dstimg, 3 * size * sizeof(BYTE));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}

	// init means
//	mean_size 


	cudaStatus = cudaMemcpy(d_label, label, size * sizeof(BYTE), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}
	cudaStatus = cudaMemcpy(d_counter, mean_counter, mean_size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}
	cudaStatus = cudaMemcpy(d_dstimg, image_dst, 3 * size * sizeof(BYTE), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}

	cudaStatus = cudaMemcpy(d_mean, mean, 3 * mean_size * sizeof(BYTE), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}

	dst = image.clone();
	cudaStatus = cudaMemcpy(d_img, image.data, 3 * size * sizeof(BYTE), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}


	initMeanWithCuda(d_mean, d_counter, d_img, mean_size, cur_means, Info, dim3(1, 1, 1), thread);
	//if (!flag) {
	//	fprintf(stderr, "set Value error -> %d %d", mean_size, cur_means);
	//}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "1. cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return -1;
	}
	//	cudaMemcpy(, d_label, size * sizeof(BYTE), cudaMemcpyDeviceToHost);

//		dim3 sizea(width, height, INIT_LINE);

	//cudaStatus = cudaMemcpy(mean, d_mean, 4 * mean_size * sizeof(BYTE), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	return -1;
	//}

	Cycle1(d_img, d_dstimg, d_mean, d_counter, d_label, Info, cur_means, dim3(1, 1, 1), thread);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "2. cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return -1;
	}

	cudaStatus = cudaMemcpy(dst.data, d_dstimg, 3 * size * sizeof(BYTE), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}

	cudaStatus = cudaMemcpy(mean, d_mean, 3 * mean_size * sizeof(BYTE), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}

	cudaStatus = cudaMemcpy(label, d_label, size * sizeof(BYTE), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}

	imshow("test", image);
	imshow("result", dst);
	waitKey(10);

	//Info.z *= 2;


	//block_width = width % block_size ? width / block_size + 1 : width / block_size;
	//block_height = height % block_size ? height / block_size + 1 : height / block_size;
	//
	//initMeanWithCuda(d_mean, d_img, mean_size, cur_means, Info, dim3(1, 1, 1), thread);
	while (1);
	return 0;

}

int k_means_video(const char* video)
{
	VideoCapture capture(video);
	Mat image;
	image = imread("ref.bmp");	
	Mat dst;
	cudaError_t cudaStatus;
	int width = //image.cols;
		(int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = //image.rows;
		(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int block_size = INIT_LINE;
	int block_width = width % block_size ? width / block_size + 1 : width / block_size;
	int block_height = height % block_size ? height / block_size + 1 : height / block_size;
	dim3 Info(width, height, block_size);
	int size = width * height;
	// 최대 블럭당 mean 개수
	int max_means = (int)sqrt(size / 2);
	// 현재 블럭당 mean 개수
	int cur_means = 5;
	//(int)sqrt(INIT_SIZE / 2);
	BYTE *mean;
	BYTE *label;
	// means의 전체 크기 -> 가장 많은 초기값으로 초기화
	int mean_size = (int)cur_means * block_width * block_height;
	// 
//	dim3 data_vec(block_size, block_size, cur_means);
	// cuda 사용할때 넣을 변수.
	// 한 쓰레드당 하나의 블럭을 수행하며, 이에 대한 블럭의 크기를 세팅해준다.
	dim3 thread(block_width, block_height);
	BYTE* d_mean;
	BYTE* d_img;
	BYTE* d_dstimg;
	BYTE* d_label;
	int* d_counter;
	BYTE* image_debug = (BYTE*)calloc(sizeof(BYTE), size * 3);

	BYTE* image_dst = (BYTE*)calloc(sizeof(BYTE), size * 3);
	int *mean_counter = (int*)calloc(sizeof(int), mean_size);
	// cur_means: init means number, height / INIT_LINE + 1: block height of max case, width / INIT_LINE + 1: block width of max case
	mean = (BYTE*)calloc(sizeof(BYTE *), 3 * mean_size);
	label = (BYTE*)calloc(sizeof(BYTE *), size);

	cudaStatus = cudaMalloc((void**)&d_mean, 3 * mean_size * sizeof(BYTE));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&d_counter, mean_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&d_label, size * sizeof(BYTE));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&d_img, 3 * size * sizeof(BYTE));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&d_dstimg, 3 * size * sizeof(BYTE));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}

	// init means
//	mean_size 


	cudaStatus = cudaMemcpy(d_label, label, size * sizeof(BYTE), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}
	cudaStatus = cudaMemcpy(d_counter, mean_counter, mean_size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}
	cudaStatus = cudaMemcpy(d_dstimg, image_dst, 3 * size * sizeof(BYTE), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}

	cudaStatus = cudaMemcpy(d_mean, mean, 3 * mean_size * sizeof(BYTE), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}

	while (1) {
		capture.read(image);
		dst = image.clone();
		cudaStatus = cudaMemcpy(d_img, image.data, 3 * size * sizeof(BYTE), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return -1;
		}

		initMeanWithCuda(d_mean, d_counter, d_img, mean_size, cur_means, Info, dim3(1, 1, 1), thread);
		//if (!flag) {
		//	fprintf(stderr, "set Value error -> %d %d", mean_size, cur_means);
		//}

		cudaMemcpy(mean,  d_mean, 3 * mean_size * sizeof(BYTE), cudaMemcpyDeviceToHost);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return -1;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "1. cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return -1;
		}
		//	cudaMemcpy(, d_label, size * sizeof(BYTE), cudaMemcpyDeviceToHost);

//		dim3 sizea(width, height, INIT_LINE);

		//cudaStatus = cudaMemcpy(mean, d_mean, 4 * mean_size * sizeof(BYTE), cudaMemcpyDeviceToHost);
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "cudaMemcpy failed!");
		//	return -1;
		//}

		Cycle1(d_img, d_dstimg, d_mean,d_counter, d_label, Info, cur_means, dim3(1, 1, 1), thread);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return -1;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "2. cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return -1;
		}

		cudaStatus = cudaMemcpy(dst.data, d_dstimg, 3 * size * sizeof(BYTE), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return -1;
		}

		cudaStatus = cudaMemcpy(mean, d_mean, 3 * mean_size * sizeof(BYTE), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return -1;
		}

		cudaStatus = cudaMemcpy(label, d_label, size * sizeof(BYTE), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return -1;
		}

		imshow("test", image);
		imshow("result", dst);
		int option;
		option = waitKey(10);

		if ((char)option == 'u') {
			printf("increase # of means: %d to %d", cur_means, cur_means + 1);
			cur_means++;
		}
		else if ((char)option == 'd') {
			printf("decrease # of means: %d to %d", cur_means, cur_means > 2 ? cur_means-- : 2);
			cur_means = cur_means > 2 ? cur_means-- : 2;
		} 
	}

	//Info.z *= 2;


	//block_width = width % block_size ? width / block_size + 1 : width / block_size;
	//block_height = height % block_size ? height / block_size + 1 : height / block_size;
	//
	//initMeanWithCuda(d_mean, d_img, mean_size, cur_means, Info, dim3(1, 1, 1), thread);
	while (1);
	return 0;
}

int main() {
//	k_means_video((const char*)"test.mp4");
	k_means_picture((const char*)"test.bmp");
}