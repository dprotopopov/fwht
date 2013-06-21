/*
Fast Walsh–Hadamard transform algorithm
Copyright (c) 2013, Dmitry Protopopov
http://protopopov.ru
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../src/fwhtcuda.cuh"


int main(int argc, char* argv[])
{
	int i;

	// Find/set the device.
	int device_count = 0;
	cudaGetDeviceCount( &device_count );
	for( i = 0 ; i < device_count ; ++i )
	{
		cudaDeviceProp properties;
		cudaGetDeviceProperties( &properties, i );
		std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
	}

	__int64 secret = 0x0123456789ABCDEF;
	int error = 80; 
	int m = 200;
	int n0 = 4;
	int n1 = 4;
	int n2 = 20;
	__int64 *a = (__int64 *)malloc( sizeof(__int64)*m );
	__int64 *b = (__int64 *)malloc( sizeof(__int64)*m );
	__int64 mask = (((__int64)1) << (n0+n1+n2)) - 1;

	int bestBufferSize = 0, worstBufferSize = 0;
	int maxBufferSize = 4;
	int * bestValueBuffer = (int *)malloc( sizeof(int)*maxBufferSize );
	__int64 * bestIndexBuffer = (__int64 *)malloc( sizeof(__int64)*maxBufferSize );
	int * worstValueBuffer = (int *)malloc( sizeof(int)*maxBufferSize );
	__int64 * worstIndexBuffer = (__int64 *)malloc( sizeof(__int64)*maxBufferSize );
	char *p;

	printf("secret:%016lX\n", secret&mask);
	printf("error:%d%%\n", error);

	printf("Generate random table...");
	srand( (unsigned)time( NULL ) );
	for( i = sizeof(__int64)*m , p = (char *)a ; i-- ; ) p[i] = (char)rand();
	for( i = m ; i-- ; ) a[i] &= mask; 
	for( i = m ; i-- ; ) 
		b[i] = fwht_cuda_parity(a[i]&secret)^(rand()%100<=error)?1:0;

	printf("Done.\n");

	printf("Find solution ...");
	time_t start = time(NULL);
	fwht_cuda_master(a, b, m, n0, n1, n2,
		bestValueBuffer, bestIndexBuffer, &bestBufferSize, maxBufferSize,
		worstValueBuffer, worstIndexBuffer, &worstBufferSize, maxBufferSize);
	time_t end = time(NULL);
	printf("Done.\n");

	printf("Results:\n");
	for(i = 0 ; i<bestBufferSize ; i++) printf("value:%8d index:%016I64X\n", bestValueBuffer[i], bestIndexBuffer[i]);
	printf("...\n");
	for(i = worstBufferSize ; i-- ; ) printf("value:%8d index:%016I64X\n", worstValueBuffer[i], worstIndexBuffer[i]);

	printf("\nExecution time: %d sec\n", (end-start));

	free( worstIndexBuffer );
	free( worstValueBuffer );
	free( bestIndexBuffer );
	free( bestValueBuffer );
	free( b );
	free( a );

	cudaDeviceReset();

	exit( 0 );
}

