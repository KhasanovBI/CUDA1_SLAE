

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

int main() {
	float timerValueCPU;
	clock_t start, stop;

	float *hA, *hX, *hX0, *hX1, *hF, *hDelta, *AT;
	float sum, eps;
	float EPS = 0.001f;
	int N = 1024 * 10;
	int size = N*N;
	int i, j, k;
	
	int Num_diag = 0.5f*(int)N*0.3f;

	unsigned int mem_sizeA = sizeof(float)*size;
	unsigned int mem_sizeX = sizeof(float)*(N);

	hA = (float*)malloc(mem_sizeA);
	AT = (float*)malloc(mem_sizeA);
	hF = (float*)malloc(mem_sizeX);
	hX = (float*)malloc(mem_sizeX);
	hX0 = (float*)malloc(mem_sizeX);
	hX1 = (float*)malloc(mem_sizeX);
	hDelta = (float*)malloc(mem_sizeX);

	for (i = 0; i<size; i++) {
		hA[i] = 0.0f;
		AT[i] = 0.0f;
	}
	// Central
	for (i = 0; i<N; i++) {
		hA[i + i*N] = rand() % 5 + 1.0f*N;
		AT[i + i*N] = hA[i + i*N];
	}
	//Up & Down 
	for (k = 1; k<Num_diag + 1; k++) {
		for (i = 0; i<N - k; i++) {
			hA[i + k + i*N] = rand() % 4;
			hA[i + (i + k)*N] = rand() % 4;
			AT[i + k + i*N] = hA[i + (i + k)*N];
			AT[i + (i + k)*N] = hA[i + k + i*N];
		}
	}
	// Convergence
	float MaxConv = 0.0f;
	for (i = 0; i<N; i++) {
		sum = 0.0f;
		for (j = 0; j<N; j++) {
			if (j != i) sum += hA[j + i*N] / hA[i + i*N];
		}
		if (sum>MaxConv) MaxConv = sum;
	}

	for (i = 0; i<N; i++) {
		hX[i] = rand() % 5;
		hX0[i] = 1.0f;
	}

	for (i = 0; i<N; i++) {
		sum = 0.0f;
		for (j = 0; j<N; j++) sum += hA[j + i*N] * hX[j];
		hF[i] = sum;
	}

	printf("\n Accuracy EPS = %f ", EPS);


	
	// CPU -------------------------------------------------------------------
	for (i = 0; i<N; i++) hX0[i] = 1.0f;

	start = clock();
	k = 0; eps = 1.0f;
	while (eps > EPS)
	{
		k++;
		for (i = 0; i<N; i++) {
			sum = 0.0f;
			for (j = 0; j<N; j++) sum += hA[j + i*N] * hX0[j];
			hX1[i] = hX0[i] + (hF[i] - sum) / hA[i + i*N];
		}
		eps = 0.0f;
		for (j = 0; j<N; j++) {
			eps += fabs(hX0[j] - hX1[j]);
			hX0[j] = hX1[j];
		}
		eps = eps / N;
		printf("\n Eps[%i]=%f ", k, eps);
	} //iter


	stop = clock();
	timerValueCPU = 1000.*(stop - start) / CLOCKS_PER_SEC;
	printf("\n CPU calculation time %f msec\n", timerValueCPU);
	free(hA);
	free(hF);
	free(hX0);
	free(hX1);
	free(hX);
	free(hDelta);

	return 0;
}