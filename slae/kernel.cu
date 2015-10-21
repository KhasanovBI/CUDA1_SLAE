#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		printf("CUDA error at: %s: %d\n", file, line);
		printf("%s %s\n", cudaGetErrorString(err), func);
		exit(1);
	}
}


__global__ void calculateSum(int N, float *deviceA, float *deviceF, float *deviceX0, float *deviceX1) {
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;	
	float sum = 0.0f;
	for (int j = 0; j < N; ++j) {
		sum += deviceA[j + i * N] * deviceX0[j];
	}
	deviceX1[i] = deviceX0[i] + (deviceF[i] - sum) / deviceA[i + i * N];
}

__global__ void calculateDifference(float *deviceX0, float *deviceX1, float *deviceDifference) {
	unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
	deviceDifference[j] = fabs(deviceX0[j] - deviceX1[j]);
	deviceX0[j] = deviceX1[j];
}


__host__ float CPUCalculation(int N, float *hA, float *hF, float *hX0, float *hX1, float EPS) {
	clock_t start, stop;
	for (int i = 0; i < N; i++) {
		hX0[i] = 1.0f;
	}
	start = clock();
	int k = 0; 
	float eps = 1.0f;
	while (eps > EPS)
	{
		k++;
		for (int i = 0; i < N; i++) {
			float sum = 0.0f;
			for (int j = 0; j < N; j++) sum += hA[j + i * N] * hX0[j];
			hX1[i] = hX0[i] + (hF[i] - sum) / hA[i + i * N];
		}
		eps = 0.0f;
		for (int j = 0; j < N; j++) {
			eps += fabs(hX0[j] - hX1[j]);
			hX0[j] = hX1[j];
		}
		eps = eps / N;
		printf("\n Eps[%i]=%f ", k, eps);
	}


	stop = clock();
	float timerValueCPU = 1000. * (stop - start) / CLOCKS_PER_SEC;

	printf("\n CPU calculation time %f msec\n", timerValueCPU);
	return timerValueCPU;
}


int main() {
	srand(time(NULL));
	
	float timerValueCPU;
	float *hA, *hX, *hX0, *hX1, *hF, *hDelta, *AT;
	float sum, eps;

	float EPS = 0.001f;
	int N = 1024 * 10;
	int size = N * N;
	int i, j, k;
	int Num_diag = 0.5f * N * 0.3f;

	unsigned int mem_sizeA = sizeof(float)*size;
	unsigned int mem_sizeX = sizeof(float)*(N);

	hA = (float*)malloc(mem_sizeA);
	AT = (float*)malloc(mem_sizeA);
	hF = (float*)malloc(mem_sizeX);
	hX = (float*)malloc(mem_sizeX);
	hX0 = (float*)malloc(mem_sizeX);
	hX1 = (float*)malloc(mem_sizeX);
	hDelta = (float*)malloc(mem_sizeX);

	for (i = 0; i < size; i++) {
		hA[i] = 0.0f;
		AT[i] = 0.0f;
	}
	// Central
	for (i = 0; i < N; i++) {
		hA[i + i * N] = rand() % 5 + 1.0f * N;
		AT[i + i * N] = hA[i + i * N];
	}
	// Up & Down 
	for (k = 1; k < Num_diag + 1; k++) {
		for (i = 0; i < N - k; i++) {
			hA[i + k + i*N] = rand() % 4;
			hA[i + (i + k)*N] = rand() % 4;
			AT[i + k + i*N] = hA[i + (i + k)*N];
			AT[i + (i + k)*N] = hA[i + k + i*N];
		}
	}
	// Convergence
	float MaxConv = 0.0f;
	for (i = 0; i < N; i++) {
		sum = 0.0f;
		for (j = 0; j<N; j++) {
			if (j != i) sum += hA[j + i*N] / hA[i + i*N];
		}
		if (sum>MaxConv) MaxConv = sum;
	}

	for (i = 0; i < N; i++) {
		hX[i] = rand() % 5;
		hX0[i] = 1.0f;
	}

	for (i = 0; i < N; i++) {
		sum = 0.0f;
		for (j = 0; j < N; j++) sum += hA[j + i*N] * hX[j];
		hF[i] = sum;
	}

	printf("\n Accuracy EPS = %f ", EPS);

	// GPU ----------------------------------------------------------------------
	float *deviceA, *deviceF, *deviceX0, *deviceX1, *deviceDifference;
	float timerValueGPU;
	cudaEvent_t GPUStart, GPUStop;
	cudaEventCreate(&GPUStart);
	cudaEventCreate(&GPUStop);


	checkCudaErrors(cudaMalloc((void**)&deviceA, mem_sizeA));
	checkCudaErrors(cudaMalloc((void**)&deviceF, mem_sizeX));
	checkCudaErrors(cudaMalloc((void**)&deviceX0, mem_sizeX));
	checkCudaErrors(cudaMalloc((void**)&deviceX1, mem_sizeX));
	checkCudaErrors(cudaMalloc((void**)&deviceDifference, mem_sizeX));
	float *hostDifference = (float *)malloc(mem_sizeX);

	checkCudaErrors(cudaMemcpy(deviceA, hA, mem_sizeA, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceF, hF, mem_sizeX, cudaMemcpyHostToDevice));
	for (i = 0; i < N; i++) {
		hX0[i] = 1.0f;
	}
	checkCudaErrors(cudaMemcpy(deviceX0, hX0, mem_sizeX, cudaMemcpyHostToDevice));

	dim3 threads(32);
	dim3 blocks(N / threads.x);

	// GPU BEGIN
	cudaEventRecord(GPUStart, 0);
	
	eps = 1.0f;
	k = 0;
	while (eps > EPS) {
		k++;
		calculateSum <<< blocks, threads >>> (N, deviceA, deviceF, deviceX0, deviceX1);
		calculateDifference <<< blocks, threads >>> (deviceX0, deviceX1, deviceDifference);
		checkCudaErrors(cudaMemcpy(hostDifference, deviceDifference, mem_sizeX, cudaMemcpyDeviceToHost));
		eps = 0.0f;
		for (j = 0; j < N; ++j) {
			eps += hostDifference[j];
		}
		eps /= N;
		printf("\n Eps[%i]=%f ", k, eps);
	}

	cudaEventRecord(GPUStop, 0);
	cudaEventSynchronize(GPUStop);
	cudaEventElapsedTime(&timerValueGPU, GPUStart, GPUStop);
	printf("\n GPU calculation time %f msec\n", timerValueGPU);
	// GPU END
	checkCudaErrors(cudaFree(deviceA));
	checkCudaErrors(cudaFree(deviceF));
	checkCudaErrors(cudaFree(deviceX0));
	checkCudaErrors(cudaFree(deviceDifference));
	free(hostDifference);




	// CPU -------------------------------------------------------------------
	timerValueCPU = CPUCalculation(N, hA, hF, hX0, hX1, EPS);
	printf("\n RATE = %f\n", timerValueCPU/timerValueGPU);

	free(hA);
	free(hF);
	free(hX0);
	free(hX1);
	free(hX);
	free(hDelta);

	getchar();
	return 0;
}