#include "kernel.h"

__device__ int getRand(curandState *s, int A, int B)
{
	float rand_int = curand_uniform(s);
	rand_int = rand_int * (B - A) + A;

	return rand_int;
}

// mean: data, img: image, mean_size: num of whole mean, cur_means: num of means per block, info: x for width, y for height, z for block interval(size)
__global__ void initMeanValue(BYTE *mean,int *counter, BYTE *img, const int mean_size, const int cur_means, dim3 Info) {
	
	int width = Info.x;
	int height = Info.y;
	int block_interval = Info.z;
	int imgsize = width * height;
	int block_number = (int)mean_size / cur_means;
	int block_per_height = height % block_interval ? (int)height / block_interval + 1 : (int)height / block_interval;
	int block_per_width = width % block_interval ? (int)width / block_interval + 1 : (int)width / block_interval;
	int block_axis_x = threadIdx.x;
	int block_axis_y = threadIdx.y;
	int temp_height = block_interval < height - block_interval * block_axis_y ? block_interval : height - block_interval * block_axis_y;
	int temp_width = block_interval < width - block_interval * block_axis_x ? block_interval : width - block_interval * block_axis_x;

	int means_interval = (int)(temp_height * temp_width - 1) / cur_means;
		//((int)block_interval*block_interval) / cur_means;
		//(sqrt((float)cur_means) + 1);
	int offset = 3 * cur_means*((int)block_per_width * block_axis_y + block_axis_x);
	int *block_count = &counter[cur_means*((int)block_per_width * block_axis_y + block_axis_x)];
	BYTE *block_mean = &mean[offset];
	int count = 0;

	for (int j = block_interval * block_axis_y; j < block_interval * (block_axis_y + 1) && j < height; j++) {
		for (int i = block_interval * block_axis_x; i < block_interval * (block_axis_x + 1) && i < width; i++) {
			if (count == cur_means)
				break;
			if ((j * block_interval + i) % means_interval == 0) {
				block_mean[3 * count] = img[3 * j * width + 3 * i];
				block_mean[3 * count + 1] = img[3 * j * width + 3 * i + 1];
				block_mean[3 * count + 2] = img[3 * j * width + 3 * i + 2];
				block_count[count] = 0;
				count++;
			}
		}
	}
	//if (count < cur_means) {
	//	for(int i = )
	//}
}


__global__ void k_means(BYTE *src, BYTE *dst, BYTE *means, int *counter, BYTE *label, dim3 size, int mean_num) {
	int width = size.x;
	int height = size.y;
	int block_size = size.z;
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int seed = id;
	curandState s;
	curand_init(seed, 0, 0, &s);

	int block_per_height = height % block_size ? (int)height / block_size + 1 : (int)height / block_size;
	int block_per_width = width % block_size ? (int)width / block_size + 1 : (int)width / block_size;
	int zero_flag = 0;
	int MPB = mean_num;
	int x_axis = threadIdx.x;
	int y_axis = threadIdx.y;
	BYTE *mean_block;
	int mean_offset = 3 * MPB * (y_axis * block_per_width + x_axis) ;

	int * block_counter = counter + MPB * (y_axis * block_per_width + x_axis);
	mean_block = &means[mean_offset];
	int rgb_sigma[100000];
		//= (unsigned int *)malloc(sizeof(unsigned int) * 3*MPB);
//	BYTE *block_mean = (BYTE*)(means + 4 * block_num * mean_count);
	int count = 0;
	int a = 0;
//	while (count < 20) {
		zero_flag = 0;
		for (int i = 0; i < MPB; i++) {
			rgb_sigma[3 * i] = 0;
			rgb_sigma[3 * i + 1] = 0;
			rgb_sigma[3 * i + 2] = 0;
			block_counter[i] = 0;
		}
		for (int j = y_axis * block_size; j < (y_axis + 1) * block_size  && j < height; j++) {
			for (int i = x_axis * block_size; i < (x_axis + 1) * block_size && i < width; i++) {

				int pixel_axis = (j * width + i);
				BYTE b = src[3*pixel_axis];
				BYTE g = src[3*pixel_axis + 1];
				BYTE r = src[3*pixel_axis + 2];
				int min = INT_MAX;
				int minIdx = 0;
				int distance;


				for (int k = 0; k < MPB; k++) {
					distance = 0;
					distance += ((int)(b - mean_block[3 * k])*(b - mean_block[3 * k]));
					distance += ((int)(g - mean_block[3 * k + 1])*(g - mean_block[3 * k + 1]));
					distance += ((int)(r - mean_block[3 * k + 2])*(r - mean_block[3 * k + 2]));
					if (min > distance) {
						min = distance;
						minIdx = k;
					}
				}

				label[pixel_axis] = minIdx;
				rgb_sigma[3 * minIdx] += (int)b;
				rgb_sigma[3 * minIdx + 1] += (int)g;
				rgb_sigma[3 * minIdx + 2] += (int)r;
				block_counter[minIdx]++;
			}
		}
		//for (int i = 0; i < MPB; i++) {
		//	if (block_counter[i] == 0 && a < 10) {
		//		mean_block[3 * i] = getRand(&s, 0, 255);
		//		mean_block[3 * i + 1] = getRand(&s, 0, 255);
		//		mean_block[3 * i + 2] = getRand(&s, 0, 255);
		//		//				mean_block[4 * i + 3] = 0;
		//		zero_flag = 1;
		//		continue;
		//	}
		//	else if (block_counter[i]== 0) {
		//		for (int j = i + 1; j < MPB; j++) {
		//			mean_block[3 * j - 3] = mean_block[3 * j];
		//			mean_block[3 * j - 2] = mean_block[3 * j + 1];
		//			mean_block[3 * j - 1] = mean_block[3 * j + 2];
		//			rgb_sigma[3 * j - 3] = rgb_sigma[3 * j];
		//			rgb_sigma[3 * j - 2] = rgb_sigma[3 * j + 1];
		//			rgb_sigma[3 * j - 1] = rgb_sigma[3 * j + 2];
		//		}
		//		zero_flag = 1;
		//		continue;
		//	}\
		//}
		//if (zero_flag) {
		//	a++;
		//	continue;
		//}
		for (int i = 0; i < MPB; i++) {
			if (block_counter[i] == 0)
				continue;
			mean_block[3 * i] = (BYTE)(rgb_sigma[3 * i] / block_counter[i]);
			mean_block[3 * i + 1] = (BYTE)(rgb_sigma[3 * i + 1] / block_counter[i]);
			mean_block[3 * i + 2] = (BYTE)(rgb_sigma[3 * i + 2] / block_counter[i]);
//			mean_block[4 * i + 3] = (BYTE)0;
				//mean_block[4 * i] = (BYTE)(rgb_sigma[3 * i] / mean_block[4 * i + 3]);
				//mean_block[4 * i + 1] = (BYTE)(rgb_sigma[3 * i + 1] / mean_block[4 * i + 3]);
				//mean_block[4 * i + 2] = (BYTE)(rgb_sigma[3 * i + 2] / mean_block[4 * i + 3]);
				//mean_block[4 * i + 3] = (BYTE)0;
		}

		count++;
		a = 0;
//	}

	for (int j = y_axis * block_size; j < (y_axis + 1) * block_size && j < height; j++) {
		for (int i = x_axis * block_size; i < (x_axis + 1) * block_size && i < width; i++) {
			int pixel_axis = (j * width + i);
			dst[3*pixel_axis] = mean_block[3*label[pixel_axis]];
			dst[3*pixel_axis + 1] = mean_block[3*label[pixel_axis] + 1];
			dst[3*pixel_axis + 2] = mean_block[3*label[pixel_axis] + 2];
		}
	}

}

void Cycle1(BYTE *src, BYTE *dst, BYTE *means,int*counter, BYTE *label, const dim3 size, int mean_num, dim3 block, dim3 thread) {
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	k_means <<<1, thread >>> (src, dst, means,counter, label, size, mean_num);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f ms\n", time);
}

void initMeanWithCuda(BYTE *mean,int *counter, BYTE *img, const int mean_size, const int cur_means, dim3 Info, dim3 block, dim3 thread) {
	initMeanValue <<<1, thread >>> (mean,counter, img, mean_size, cur_means, Info);
}