/*
Fast Walsh–Hadamard transform algorithm
Copyright (c) 2013, Dmitry Protopopov
http://protopopov.ru
*/

#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define FWHT_CUDA_STACK_SIZE 64

template< class VALUE, class INDEX > 
__global__ void fwht_cuda_transform_default_wrapper(VALUE * v, INDEX count, INDEX blockSize)
{
	INDEX i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i*blockSize < count) fwht_cuda_transform_default(&v[i*blockSize],blockSize);
}

template< class VALUE, class INDEX > 
__global__ void fwht_cuda_transform_block_wrapper(VALUE * v, INDEX count, INDEX blockSize) 
{
	INDEX i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < blockSize) fwht_cuda_transform_block(&v[i],count,blockSize);
}

template< class VALUE, class INDEX > 
__device__ void fwht_cuda_transform_default(VALUE * v, INDEX count) 
{
	VALUE * stackPtrValue[FWHT_CUDA_STACK_SIZE];
	INDEX stackIndex[FWHT_CUDA_STACK_SIZE];
	int stackSize = 0;

	while(true) {
		while (count > 1) {
			count >>= 1;

			VALUE t1;
			VALUE t2;
			VALUE * v1 = v;
			VALUE * v2 = &v[count];

			for (INDEX i = count ; i-- ; ) {
				t1 = *v1 ;
				t2 = *v2;
				*v1++ = t1 + t2;
				*v2++ = t1 - t2;
			}
			stackPtrValue[stackSize] = v1;
			stackIndex[stackSize] = count;
			stackSize++;
		}
		if(stackSize) {
			stackSize--;
			v = stackPtrValue[stackSize];
			count = stackIndex[stackSize];
		} else {
			break;
		}
	}

}

template< class VALUE, class INDEX > 
__device__ void fwht_cuda_transform_block(VALUE * v, INDEX count, INDEX blockSize) 
{
	VALUE * stackPtrValue[FWHT_CUDA_STACK_SIZE];
	INDEX stackIndex[FWHT_CUDA_STACK_SIZE];
	int stackSize = 0;

	while(true) {
		while (count > blockSize) {
			count >>= 1;

			VALUE t1;
			VALUE t2;
			VALUE * v1 = v;
			VALUE * v2 = &v[count];

			for (INDEX i = 0 ; i < count ; i += blockSize) {
				t1 = *v1 ;
				t2 = *v2;
				*v1 = t1 + t2;
				*v2 = t1 - t2;
				v1 = &v1[blockSize];
				v2 = &v2[blockSize];
			}
			stackPtrValue[stackSize] = v1;
			stackIndex[stackSize] = count;
			stackSize++;
		}
		if(stackSize) {
			stackSize--;
			v = stackPtrValue[stackSize];
			count = stackIndex[stackSize];
		} else {
			break;
		}
	}
}


template< class VALUE, class INDEX > 
__host__ void fwht_cuda_worker(
	INDEX *a, INDEX *b, VALUE m, 
	int n1, int n2, 
	VALUE *bestValueBuffer, INDEX *bestIndexBuffer, int * pBestBufferSize, int maxBestBufferSize,
	VALUE *worstValueBuffer, INDEX *worstIndexBuffer, int * pWorstBufferSize, int maxWorstBufferSize) 
{
	cudaError_t err;
	VALUE * h_v = NULL;
	VALUE * d_v = NULL;

	err = cudaMalloc(&d_v, sizeof(VALUE) << n2 );
	h_v = (VALUE *)malloc( sizeof(VALUE) << n2 );
	assert (err == cudaSuccess);
	assert (h_v != NULL);

	INDEX count1 = ((INDEX)1) << n1;
	INDEX count2 = ((INDEX)1) << n2;
	INDEX mask2 = count2 - 1;
	int n20 = n2>>1;
	int n21 = n2 - n20;
	INDEX count20 = 1 << n20;
	INDEX count21 = 1 << n21;

	for(INDEX indexBase = 0; count1--; indexBase += count2) {
		memset(h_v,0,sizeof(VALUE)<<n2);

		for(VALUE i = m ; i-- ; ) {
			if (fwht_cuda_parity(a[i]&indexBase^b[i]))
				h_v[a[i]&mask2]--;
			else
				h_v[a[i]&mask2]++;
		}

		err = cudaMemcpy(d_v, h_v, sizeof(VALUE)*count2, cudaMemcpyHostToDevice);
		assert (err == cudaSuccess);

		/////////////////////////////////////////////////
		// The way to parallel between threads
		switch (count2) {
		case 0:
		case 1:
			break;
		case 2:
			fwht_cuda_transform_default_wrapper<<<1,1>>>(d_v,count2,count2);
			break;
		default:
			fwht_cuda_transform_block_wrapper<<<(count20+255)/256, 256>>>(d_v, count2, count21);
			fwht_cuda_transform_default_wrapper<<<(count20+255)/256, 256>>>(d_v, count2, count21);
			break;
		}
		//////////////////////////////////////////////////

		err = cudaMemcpy(h_v, d_v, sizeof(VALUE)*count2, cudaMemcpyDeviceToHost);
		assert (err == cudaSuccess);

		fwht_cuda_get_best_indexes(h_v,count2,indexBase,bestValueBuffer,bestIndexBuffer,pBestBufferSize,maxBestBufferSize);
		fwht_cuda_get_worst_indexes(h_v,count2,indexBase,worstValueBuffer,worstIndexBuffer,pWorstBufferSize,maxWorstBufferSize);
	}

	cudaFree(d_v);
	free(h_v);
}


template< class VALUE, class INDEX > 
__host__ void fwht_cuda_master(
	INDEX *a, INDEX *b, VALUE m, 
	int n0, int n1, int n2, 
	VALUE *bestValueBuffer, INDEX *bestIndexBuffer, int * pBestBufferSize, int maxBestBufferSize,
	VALUE *worstValueBuffer, INDEX *worstIndexBuffer, int * pWorstBufferSize, int maxWorstBufferSize) 
{
	INDEX count0 = ((INDEX)1) << n0;
	INDEX count12 = ((INDEX)1) << (n1+n2);
	INDEX mask = count12-1;

	for(INDEX indexBase = 0; count0--; indexBase += count12) {
		INDEX * workerA = (INDEX *)malloc(sizeof(INDEX)*m);
		INDEX * workerB = (INDEX *)malloc(sizeof(INDEX)*m);
		VALUE * workerBestValueBuffer = (VALUE *)malloc(sizeof(VALUE)*maxBestBufferSize);
		INDEX * workerBestIndexBuffer = (INDEX *)malloc(sizeof(INDEX)*maxBestBufferSize);
		int workerBestBufferSize = 0;
		VALUE * workerWorstValueBuffer = (VALUE *)malloc(sizeof(VALUE)*maxWorstBufferSize);
		INDEX * workerWorstIndexBuffer = (INDEX *)malloc(sizeof(INDEX)*maxWorstBufferSize);

		assert (workerA != NULL);
		assert (workerB != NULL);
		assert (workerBestValueBuffer != NULL);
		assert (workerBestIndexBuffer != NULL);
		assert (workerWorstValueBuffer != NULL);
		assert (workerWorstIndexBuffer != NULL);

		int workerWorstBufferSize = 0;
		for(VALUE i = m ; i-- ; ) {
			workerA[i] = a[i]&mask;
			workerB[i] = a[i]&indexBase^b[i];
		}
		fwht_cuda_worker(workerA, workerB, m, n1, n2,
			workerBestValueBuffer, workerBestIndexBuffer, &workerBestBufferSize, maxBestBufferSize,
			workerWorstValueBuffer, workerWorstIndexBuffer, &workerWorstBufferSize, maxWorstBufferSize);
		fwht_cuda_sort_desc(bestValueBuffer,bestIndexBuffer,pBestBufferSize,maxBestBufferSize,
			indexBase,workerBestValueBuffer,workerBestIndexBuffer,workerBestBufferSize);
		fwht_cuda_sort_asc(worstValueBuffer,worstIndexBuffer,pWorstBufferSize,maxWorstBufferSize,
			indexBase,workerWorstValueBuffer,workerWorstIndexBuffer,workerWorstBufferSize);

		free(workerWorstIndexBuffer);
		free(workerWorstValueBuffer);
		free(workerBestIndexBuffer);
		free(workerBestValueBuffer);
		free(workerB);
		free(workerA);
	}
}

template< class VALUE, class INDEX > 
__host__ void fwht_cuda_get_best_indexes(
	VALUE * v, INDEX count, 
	INDEX indexBase, 
	VALUE *valueBuffer, INDEX *indexBuffer, int * pBufferSize, int maxBufferSize
	) 
{
	while(count--) {
		VALUE t = v[count];
		int i = *pBufferSize;
		while ( i-- && valueBuffer[i] < t ) ;
		if( ++i < maxBufferSize) {
			if (*pBufferSize < maxBufferSize) (*pBufferSize)++;
			int j = (*pBufferSize  < maxBufferSize) ? *pBufferSize : maxBufferSize;
			while ( --j > i ) {
				valueBuffer[j] = valueBuffer[j-1];
				indexBuffer[j] = indexBuffer[j-1];
			}
			valueBuffer[i] = t;
			indexBuffer[i] = count ^ indexBase;
		}
	}
}

template< class VALUE, class INDEX > 
__host__ void fwht_cuda_get_worst_indexes(
	VALUE * v, INDEX count, 
	INDEX indexBase, 
	VALUE *valueBuffer, INDEX *indexBuffer, int * pBufferSize, int maxBufferSize
	) 
{
	while(count--) {
		VALUE t = v[count];
		int i = *pBufferSize;
		while ( i-- && valueBuffer[i] > t ) ;
		if( ++i < maxBufferSize) {
			if (*pBufferSize < maxBufferSize) (*pBufferSize)++;
			int j = (*pBufferSize < maxBufferSize) ? *pBufferSize  : maxBufferSize;
			while ( --j > i ) {
				valueBuffer[j] = valueBuffer[j-1];
				indexBuffer[j] = indexBuffer[j-1];
			}
			valueBuffer[i] = t;
			indexBuffer[i] = count ^ indexBase;
		}
	}
}

template< class VALUE, class INDEX> 
__host__ void fwht_cuda_sort_desc(
	VALUE *valueBuffer, INDEX *indexBuffer, int * pBufferSize, int maxBufferSize,
	INDEX indexBase, VALUE *valueBufferAdd, INDEX *indexBufferAdd, int addBufferSize
	)
{
	VALUE * pTempValueBuffer = (VALUE *)malloc(sizeof(VALUE)*maxBufferSize);
	INDEX * pTempIndexBuffer = (INDEX *)malloc(sizeof(INDEX)*maxBufferSize);
	int tempBufferSize = 0;
	int i = 0 , j = 0;

	assert (pTempValueBuffer != NULL);
	assert (pTempIndexBuffer != NULL);

	while (i<*pBufferSize && j<addBufferSize && tempBufferSize<maxBufferSize) {
		if (valueBuffer[i]<valueBufferAdd[j]) {
			pTempValueBuffer[tempBufferSize] = valueBufferAdd[j];
			pTempIndexBuffer[tempBufferSize] = indexBufferAdd[j]^indexBase;
			j++; tempBufferSize++;
		} else {
			pTempValueBuffer[tempBufferSize] = valueBuffer[i];
			pTempIndexBuffer[tempBufferSize] = indexBuffer[i];
			i++; tempBufferSize++;
		}
	}
	while (i<*pBufferSize  && tempBufferSize<maxBufferSize) {
		pTempValueBuffer[tempBufferSize] = valueBuffer[i];
		pTempIndexBuffer[tempBufferSize] = indexBuffer[i];
		i++; tempBufferSize++;
	}
	while (j<addBufferSize && tempBufferSize<maxBufferSize) {
		pTempValueBuffer[tempBufferSize] = valueBufferAdd[j];
		pTempIndexBuffer[tempBufferSize] = indexBufferAdd[j]^indexBase;
		j++; tempBufferSize++;
	}
	memcpy(valueBuffer,pTempValueBuffer,sizeof(VALUE)*tempBufferSize);
	memcpy(indexBuffer,pTempIndexBuffer,sizeof(INDEX)*tempBufferSize);
	*pBufferSize = tempBufferSize;
	free(pTempValueBuffer);
	free(pTempIndexBuffer);
}

template< class VALUE, class INDEX> 
__host__ void fwht_cuda_sort_asc(
	VALUE *valueBuffer, INDEX *indexBuffer, int * pBufferSize, 	int maxBufferSize,
	INDEX indexBase, VALUE *valueBufferAdd, INDEX *indexBufferAdd, int addBufferSize
	)
{
	VALUE * pTempValueBuffer = (VALUE *)malloc(sizeof(VALUE)*maxBufferSize);
	INDEX * pTempIndexBuffer = (INDEX *)malloc(sizeof(INDEX)*maxBufferSize);
	int tempBufferSize = 0;
	int i = 0 , j = 0;

	assert (pTempValueBuffer != NULL);
	assert (pTempIndexBuffer != NULL);

	while (i<*pBufferSize && j<addBufferSize && tempBufferSize<maxBufferSize) {
		if (valueBuffer[i]>valueBufferAdd[j]) {
			pTempValueBuffer[tempBufferSize] = valueBufferAdd[j];
			pTempIndexBuffer[tempBufferSize] = indexBufferAdd[j]^indexBase;
			j++; tempBufferSize++;
		} else {
			pTempValueBuffer[tempBufferSize] = valueBuffer[i];
			pTempIndexBuffer[tempBufferSize] = indexBuffer[i];
			i++; tempBufferSize++;
		}
	}
	while (i<*pBufferSize  && tempBufferSize<maxBufferSize) {
		pTempValueBuffer[tempBufferSize] = valueBuffer[i];
		pTempIndexBuffer[tempBufferSize] = indexBuffer[i];
		i++; tempBufferSize++;
	}
	while (j<addBufferSize && tempBufferSize<maxBufferSize) {
		pTempValueBuffer[tempBufferSize] = valueBufferAdd[j];
		pTempIndexBuffer[tempBufferSize] = indexBufferAdd[j]^indexBase;
		j++; tempBufferSize++;
	}
	memcpy(valueBuffer,pTempValueBuffer,sizeof(VALUE)*tempBufferSize);
	memcpy(indexBuffer,pTempIndexBuffer,sizeof(INDEX)*tempBufferSize);
	*pBufferSize = tempBufferSize;
	free(pTempValueBuffer);
	free(pTempIndexBuffer);
}

template< class INDEX > 
__host__ int fwht_cuda_parity(INDEX t) {
	for(int i = sizeof(INDEX)<<2 ; i ; i >>= 1) t ^= t >> i;
	return t&1;
}

