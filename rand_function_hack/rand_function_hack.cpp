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
	int m = 1000;
	int n0 = 0;
	int n1 = 0;
	int n2 = 16;

	__int64 *srand_argument = new __int64[m];
	__int64 *rand_function = new __int64[m];

	__int64 *a = new __int64[m];
	__int64 *b = new __int64[m];
	__int64 mask = (((__int64)1) << (n0+n1+n2)) - 1;

	__int64 *s2rA = new __int64[n0+n1+n2];
	__int64 s2rB = 0;

	__int64 *r2sA = new __int64[n0+n1+n2];
	__int64 r2sB = 0;

	__int64 *s2sA = new __int64[n0+n1+n2];
	__int64 s2sB = 0;

	int bestBufferSize, worstBufferSize;
	int maxBufferSize = 1;
	int * bestValueBuffer = new int[maxBufferSize];
	__int64 * bestIndexBuffer = new __int64[maxBufferSize];
	int * worstValueBuffer = new int[maxBufferSize];
	__int64 * worstIndexBuffer = new __int64[maxBufferSize];
	
	printf("Collect rand function values ... ");
	for(int i = 0 ; i < m ; i++ ) {
		__int64 arg_of_srand = (time(NULL)^i^rand()) & mask;
		srand((unsigned)arg_of_srand);
		__int64 rand_func = rand() & mask;
		srand_argument[i] = arg_of_srand;
		rand_function[i] = rand_func ;
	}
	printf("Done.\n");

	printf("Detect rand function value from srand argument\n");
	for(int i = 0 ; i < n0 + n1 + n2 ; i++ ) {
		// printf("Detect linear function for bit #%d ... ", i);
	
		bestBufferSize = 0;
		worstBufferSize = 0;
		__int64 bit = ((__int64)1) << i ;

		for(int j = 0 ; j < m ; j++ ) {
			a[j] = srand_argument[j] & mask;
			b[j] = rand_function[j] & bit;
		}

		fwht_master(a, b, m, n0, n1, n2,
			bestValueBuffer, bestIndexBuffer, &bestBufferSize, maxBufferSize,
			worstValueBuffer, worstIndexBuffer, &worstBufferSize, maxBufferSize);

		if (bestValueBuffer[0] > -worstValueBuffer[0] ) {
			s2rA[i] = bestIndexBuffer[0] ;
		} else {
			s2rA[i] = worstIndexBuffer[0] ;
			s2rB ^= bit ;
		}
		// printf("Done.\n");
	}
	printf("Results:\n");
	for(int i = 0 ; i < n0+n1+n2 ; i++) printf("A[%02d] = %016I64X\n", i, s2rA[i]); printf("B     = %016I64X\n", s2rB);
	printf("Note: rand_func = A & arg_of_srand ^ B\n");

	printf("Testing ... ");
	int numTests = 10000;
	float avgDistance = 0.0f;
	for(int i = 0; i < numTests ; i++ ) {
		__int64 arg_of_srand = (time(NULL)^i^rand()) & mask;
		srand((unsigned)arg_of_srand);
		__int64 rand_func = rand() & mask;
		__int64 predict = s2rB;
		for(int k = 0 ; k < n0+n1+n2 ; k++) {
			predict ^= ((__int64)fwht_parity(s2rA[k]&arg_of_srand)) << k;
		}
		__int64 diff = rand_func ^ predict;
		int distance = 0;
		for(int k = 0 ; k < n0+n1+n2 ; k++) {
			distance += (diff>>k)&1;
		}
		avgDistance += (float)distance;
	}
	avgDistance /= (float)numTests;
	printf("Done.\n");

	printf("\nAverage distance: %f of %d bits.\n\n", avgDistance, n0+n1+n2);
	
	printf("Detect srand argument from rand function value\n");
	for(int i = 0 ; i < n0 + n1 + n2 ; i++ ) {
		// printf("Detect linear function for bit #%d ... ", i);
	
		bestBufferSize = 0;
		worstBufferSize = 0;
		__int64 bit = ((__int64)1) << i ;

		for(int j = 0 ; j < m ; j++ ) {
			a[j] = rand_function[j] & mask;
			b[j] = srand_argument[j] & bit;
		}

		fwht_master(a, b, m, n0, n1, n2,
			bestValueBuffer, bestIndexBuffer, &bestBufferSize, maxBufferSize,
			worstValueBuffer, worstIndexBuffer, &worstBufferSize, maxBufferSize);

		if (bestValueBuffer[0] > -worstValueBuffer[0] ) {
			r2sA[i] = bestIndexBuffer[0] ;
		} else {
			r2sA[i] = worstIndexBuffer[0] ;
			r2sB ^= bit ;
		}
		// printf("Done.\n");
	}
	printf("Results:\n");
	for(int i = 0 ; i < n0+n1+n2 ; i++) printf("A[%02d] = %016I64X\n", i, r2sA[i]); printf("B     = %016I64X\n", r2sB);
	printf("Note: arg_of_srand = A & rand_func ^ B\n\n");


	printf("Detect next srand argument from previous srand argument\n");
	srand((unsigned)time(NULL));

	__int64 * srand_values = new __int64[m+1];
	for(int i = 0 ; i < m+1 ; i++ ) {
		__int64 rand_func = rand()&mask;
		__int64 arg_of_srand = r2sB;
		for(int k = 0 ; k < n0+n1+n2 ; k++) {
			arg_of_srand ^= ((__int64)fwht_parity(r2sA[k]&rand_func)) << k;
		}
		srand_values[i] = arg_of_srand;
	}
	for(int i = 0 ; i < n0 + n1 + n2 ; i++ ) {
		// printf("Detect linear function for bit #%d ... ", i);
	
		bestBufferSize = 0;
		worstBufferSize = 0;
		__int64 bit = ((__int64)1) << i ;

		for(int j = 0 ; j < m ; j++ ) {
			a[j] = srand_values[j] & mask;
			b[j] = srand_values[j+1] & bit;
		}

		fwht_master(a, b, m, n0, n1, n2,
			bestValueBuffer, bestIndexBuffer, &bestBufferSize, maxBufferSize,
			worstValueBuffer, worstIndexBuffer, &worstBufferSize, maxBufferSize);

		if (bestValueBuffer[0] > -worstValueBuffer[0] ) {
			s2sA[i] = bestIndexBuffer[0] ;
		} else {
			s2sA[i] = worstIndexBuffer[0] ;
			s2sB ^= bit ;
		}
		// printf("Done.\n");
	}
	printf("Results:\n");
	for(int i = 0 ; i < n0+n1+n2 ; i++) printf("A[%02d] = %016I64X\n", i, s2sA[i]); printf("B     = %016I64X\n", s2sB);
	printf("Note: next_arg_of_srand = A & prev_arg_of_srand ^ B\n\n");


	printf("Detect srand argument from rand function sequence\n");
	int m2 = 1000;
	__int64 **rn2sA = (__int64 **)malloc(sizeof(__int64*)*m2);
	for(int n = 0 ; n < m2 ; n++) rn2sA[n] = (__int64 *)malloc(sizeof(__int64)*(n0+n1+n2));
	__int64 *rn2sB = (__int64 *)malloc(sizeof(__int64)*m2);

	
	printf("Build linear systems ");
	for(int n = 0 ; n < m2 ; n++ ) {
		printf(".");
		for(int i = 0 ; i < m ; i++ ) {
			__int64 arg_of_srand = (time(NULL)^i^rand()) & mask;
			srand((unsigned)arg_of_srand);
			__int64 rand_func = rand() & mask;
			for(int k = 0 ; k < n ; k++) rand_func = rand() & mask;
			srand_argument[i] = arg_of_srand;
			rand_function[i] = rand_func ;
		}

		rn2sB[n] = 0;
		for(int i = 0 ; i < n0 + n1 + n2 ; i++ ) {
			bestBufferSize = 0;
			worstBufferSize = 0;
			__int64 bit = ((__int64)1) << i ;

			for(int j = 0 ; j < m ; j++ ) {
				a[j] = rand_function[j] & mask;
				b[j] = srand_argument[j] & bit;
			}

			fwht_master(a, b, m, n0, n1, n2,
				bestValueBuffer, bestIndexBuffer, &bestBufferSize, maxBufferSize,
				worstValueBuffer, worstIndexBuffer, &worstBufferSize, maxBufferSize);

			if (bestValueBuffer[0] > -worstValueBuffer[0] ) {
				rn2sA[n][i] = bestIndexBuffer[0] ;
			} else {
				rn2sA[n][i] = worstIndexBuffer[0] ;
				rn2sB[n] ^= bit ;
			}
		}
	}
	printf(" Done.\n");

	__int64 secret = 0x0123456789ABCDEF & mask;
	int *vote = (int *)malloc(sizeof(int)*(n0+n1+n2));
	memset(vote, 0 , sizeof(int)*(n0+n1+n2));
	printf("Secret: %016I64X\n", secret);

	srand((unsigned)secret);

	printf("Calculate and vote ... ");
	for(int n = 0 ; n < m2 ; n++ ) {
		__int64 rand_func = rand();
		__int64 arg_of_srand = rn2sB[n];
		for(int k = 0 ; k < n0+n1+n2 ; k++) {
			arg_of_srand ^= ((__int64)fwht_parity(rn2sA[n][k]&rand_func)) << k;
		}
		for(int k = 0 ; k < n0+n1+n2 ; k++) {
			if ((arg_of_srand >> k) & 1) {
				vote[k]++;
			} else {
				vote[k]--;
			}
		}
	}
	__int64 solution = 0;
	for(int k = 0 ; k < n0+n1+n2 ; k++) {
		if (vote[k] > 0) {
			solution ^= ((__int64)1) << k;
		}
	}
	printf("Done.\n");

	printf("\nSolution: %016I64X\n", solution);

	free( vote );
	free( rn2sB );
	for(int n = 0 ; n < m2 ; n++) free( rn2sA[n] );
	free( rn2sA );

	delete worstIndexBuffer;
	delete worstValueBuffer;
	delete bestIndexBuffer;
	delete bestValueBuffer;
	delete b;
	delete a;
	delete s2rA;
	delete r2sA;
	delete s2sA;
	delete srand_values;
	delete srand_argument;
	delete rand_function;

	return 0;
}

