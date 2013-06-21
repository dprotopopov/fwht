/*
Fast Walsh–Hadamard transform algorithm
Copyright (c) 2013, Dmitry Protopopov
http://protopopov.ru
*/

#include <stdlib.h>
#include <memory.h>
#include <assert.h>


template< class VALUE, class INDEX > 
void fwht_transform_default(VALUE * v, INDEX count) 
{
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
		fwht_transform_default(v1, count);
	}
}

template< class VALUE, class INDEX > 
void fwht_transform_block(VALUE * v, INDEX count, INDEX blockSize) 
{
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
		fwht_transform_block(v1, count, blockSize);
	}
}

template< class VALUE, class INDEX > 
void fwht_worker(
	INDEX *a, INDEX *b, VALUE m, 
	int n1, int n2, 
	VALUE *pBestValueBuffer, INDEX *pBestIndexBuffer, int * pBestempBufferSize, int maxBestBufferSize,
	VALUE *pWorstValueBuffer, INDEX *pWorstIndexBuffer, int * pWorstBufferSize, int maxWorstBufferSize
	) 
{

	VALUE * v = (VALUE *)malloc( sizeof(VALUE) << n2 );
	assert (v != NULL);

	INDEX count1 = ((INDEX)1) << n1;
	INDEX count2 = ((INDEX)1) << n2;
	INDEX mask2 = count2 - 1;

	for(INDEX indexBase = 0; count1--; indexBase += count2) {
		memset(v,0,sizeof(VALUE)<<n2);
		for(VALUE i = m ; i-- ; ) {
			if (fwht_parity(a[i]&indexBase^b[i])) {
				v[a[i]&mask2]--;
			} else {
				v[a[i]&mask2]++;
			}
		}
		/////////////////////////////////////////////////
		// The way to parallel between threads
		switch (count2) {
		case 0:
		case 1:
			break;
		case 2:
			fwht_transform_default(v,count2);
			break;
		default:
			INDEX blockSize = ((INDEX)1) << (n2 >> 1);
			for(INDEX i = blockSize; i-- ; ) {
				fwht_transform_block(&v[i], count2, blockSize);
			}
			for(INDEX i = 0; i < count2 ; i += blockSize) {
				fwht_transform_default(&v[i], blockSize);
			}
			break;
		}
		//////////////////////////////////////////////////
		fwht_get_best_indexes(v,count2,indexBase,pBestValueBuffer,pBestIndexBuffer,pBestempBufferSize,maxBestBufferSize);
		fwht_get_worst_indexes(v,count2,indexBase,pWorstValueBuffer,pWorstIndexBuffer,pWorstBufferSize,maxWorstBufferSize);
	}

	free( v );
}


template< class VALUE, class INDEX > 
void fwht_master(
	INDEX *a, INDEX *b, VALUE m, 
	int n0, int n1, int n2, 
	VALUE *pBestValueBuffer, INDEX *pBestIndexBuffer, int * pBestempBufferSize, int maxBestBufferSize,
	VALUE *pWorstValueBuffer, INDEX *pWorstIndexBuffer, int * pWorstBufferSize, int maxWorstBufferSize
	) 
{
	INDEX count0 = ((INDEX)1) << n0;
	INDEX count12 = ((INDEX)1) << (n1+n2);
	INDEX mask = count12-1;

	////////////////////////////////////////////////////////////
	// The way to parallel between devices 
	for(INDEX indexBase = 0; count0--; indexBase += count12) {
		INDEX * workerA = (INDEX *)malloc(sizeof(INDEX)*m);
		INDEX * workerB = (INDEX *)malloc(sizeof(INDEX)*m);
		VALUE * pWorkerBestValueBuffer = (VALUE *)malloc(sizeof(VALUE)*maxBestBufferSize);
		INDEX * pWorkerBestIndexBuffer = (INDEX *)malloc(sizeof(INDEX)*maxBestBufferSize);
		int workerBestBufferSize = 0;
		VALUE * pWorkerWorstValueBuffer = (VALUE *)malloc(sizeof(VALUE)*maxWorstBufferSize);
		INDEX * pWorkerWorstIndexBuffer = (INDEX *)malloc(sizeof(INDEX)*maxWorstBufferSize);
		int workerWorstBufferSize = 0;

		assert (workerA != NULL);
		assert (workerB != NULL);
		assert (workerBestValueBuffer != NULL);
		assert (workerBestIndexBuffer != NULL);
		assert (workerWorstValueBuffer != NULL);
		assert (workerWorstIndexBuffer != NULL);

		for(VALUE i = m ; i-- ; ) {
			workerA[i] = a[i]&mask;
			workerB[i] = a[i]&indexBase^b[i];
		}
		
		fwht_worker(workerA, workerB, m, n1, n2,
			pWorkerBestValueBuffer, pWorkerBestIndexBuffer, &workerBestBufferSize, maxBestBufferSize,
			pWorkerWorstValueBuffer, pWorkerWorstIndexBuffer, &workerWorstBufferSize, maxWorstBufferSize);
		fwht_sort_desc(pBestValueBuffer,pBestIndexBuffer,pBestempBufferSize,maxBestBufferSize,
			indexBase,pWorkerBestValueBuffer,pWorkerBestIndexBuffer,workerBestBufferSize);
		fwht_sort_asc(pWorstValueBuffer,pWorstIndexBuffer,pWorstBufferSize,maxWorstBufferSize,
			indexBase,pWorkerWorstValueBuffer,pWorkerWorstIndexBuffer,workerWorstBufferSize);

		free(pWorkerWorstIndexBuffer);
		free(pWorkerWorstValueBuffer);
		free(pWorkerBestIndexBuffer);
		free(pWorkerBestValueBuffer);
		free(workerB);
		free(workerA);
	}
	////////////////////////////////////////////////////////
}

template< class VALUE, class INDEX > 
void fwht_get_best_indexes(
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
void fwht_get_worst_indexes(
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
void fwht_sort_desc(
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
void fwht_sort_asc(
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
int fwht_parity(INDEX t) {
	for(int i = sizeof(INDEX)<<2 ; i ; i >>= 1) t ^= t >> i;
	return t&1;
}

