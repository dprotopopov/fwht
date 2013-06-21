/*
Fast Walsh–Hadamard transform algorithm
Copyright (c) 2013, Dmitry Protopopov
http://protopopov.ru
*/

#include "stdafx.h"
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "../src/fwht.h"


int main(int argc, char* argv[])
{
	int i;

	__int64 secret = 0x0123456789ABCDEF;
	int error = 80; 
	int m = 200;
	int n0 = 4;
	int n1 = 4;
	int n2 = 20;
	__int64 *a = new __int64[m];
	__int64 *b = new __int64[m];
	__int64 mask = (((__int64)1) << (n0+n1+n2)) - 1;
	
	int bestBufferSize = 0, worstBufferSize = 0;
	int maxBufferSize = 4;
	int * bestValueBuffer = new int[maxBufferSize];
	__int64 * bestIndexBuffer = new __int64[maxBufferSize];
	int * worstValueBuffer = new int[maxBufferSize];
	__int64 * worstIndexBuffer = new __int64[maxBufferSize];
	char *p;

	printf("secret:%016lX\n", secret&mask);
	printf("error:%d%%\n", error);

	printf("Generate random table...");
	srand( (unsigned)time( NULL ) );
	for( i = sizeof(__int64)*m , p = (char *)a ; i-- ; ) p[i] = (char)rand();
	for( i = m ; i-- ; ) a[i] &= mask; 
	for( i = m ; i-- ; ) 
			b[i] = fwht_parity(a[i]&secret)^(rand()%100<=error)?1:0;

	printf("Done.\n");

	printf("Find solution ...");
	time_t start = time(NULL);
	fwht_master(a, b, m, n0, n1, n2,
		bestValueBuffer, bestIndexBuffer, &bestBufferSize, maxBufferSize,
		worstValueBuffer, worstIndexBuffer, &worstBufferSize, maxBufferSize);
	time_t end = time(NULL);
	printf("Done.\n");

	printf("Results:\n");
	for(i = 0 ; i<bestBufferSize ; i++) printf("value:%8d index:%016I64X\n", bestValueBuffer[i], bestIndexBuffer[i]);
	printf("...\n");
	for(i = worstBufferSize ; i-- ; ) printf("value:%8d index:%016I64X\n", worstValueBuffer[i], worstIndexBuffer[i]);

	printf("\nExecution time: %d sec\n", (end-start));

	delete worstIndexBuffer;
	delete worstValueBuffer;
	delete bestIndexBuffer;
	delete bestValueBuffer;
	delete b;
	delete a;

	return 0;
}

