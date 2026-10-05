/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten/emscripten.h>
#include "blis.h"
#include "cblas.h"

#if defined(BENCH_CGEMM)
typedef scomplex scalar_t;
#define BENCH_DT BLIS_SCOMPLEX
#define BENCH_COMPLEX 1
#define GEMM(a, b, c) cblas_cgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, &one, a, n, b, n, &zero, c, n )
#elif defined(BENCH_ZGEMM)
typedef dcomplex scalar_t;
#define BENCH_DT BLIS_DCOMPLEX
#define BENCH_COMPLEX 1
#define GEMM(a, b, c) cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, &one, a, n, b, n, &zero, c, n )
#else
typedef float scalar_t;
#define BENCH_DT BLIS_FLOAT
#define BENCH_COMPLEX 0
#define GEMM(a, b, c) cblas_sgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, one, a, n, b, n, zero, c, n )
#endif

typedef struct { double r, i; } value_t;

static value_t get( scalar_t x )
{
#if BENCH_COMPLEX
	return ( value_t ){ x.real, x.imag };
#else
	return ( value_t ){ x, 0 };
#endif
}

static scalar_t put( double r, double i )
{
#if BENCH_COMPLEX
	return ( scalar_t ){ r, i };
#else
	(void)i;
	return r;
#endif
}

int main( int argc, char** argv )
{
	if ( argc != 4 )
	{
		fprintf( stderr, "usage: bench-gemm.js SIZE native|1m|default REPS (0 to calibrate)\n" );
		return 2;
	}
	const int n = atoi( argv[1] );
	int reps = atoi( argv[3] );
	if ( n < 2 || n > 4096 || reps < 0 ) return 2;
	bli_init();
	if ( !strcmp( argv[2], "native" ) )
		bli_ind_oper_enable_only( BLIS_GEMM, BLIS_NAT, BENCH_DT );
	else if ( BENCH_COMPLEX && !strcmp( argv[2], "1m" ) )
		bli_ind_oper_enable_only( BLIS_GEMM, BLIS_1M, BENCH_DT );
	else if ( strcmp( argv[2], "default" ) ) return 2;
	const ind_t method = bli_ind_oper_find_avail( BLIS_GEMM, BENCH_DT );
	fprintf( stderr, "method=%s\n", method == BLIS_NAT ? "native" : method == BLIS_1M ? "1m" : "other" );
	if ( !strcmp( argv[2], "native" ) && method != BLIS_NAT ) return 1;
	if ( !strcmp( argv[2], "1m" ) && method != BLIS_1M ) return 1;

	const size_t count = ( size_t )n * n;
	scalar_t* a = malloc( count * sizeof( scalar_t ) );
	scalar_t* b = malloc( count * sizeof( scalar_t ) );
	scalar_t* c = malloc( count * sizeof( scalar_t ) );
	if ( !a || !b || !c ) return 1;
	for ( size_t j = 0; j < count; ++j )
	{
		a[j] = put( ( ( int )( j*13 % 17 ) - 8 ) * 0.25, ( ( int )( j*7 % 13 ) - 6 ) * 0.25 );
		b[j] = put( ( ( int )( j*11 % 19 ) - 9 ) * 0.25, ( ( int )( j*5 % 11 ) - 5 ) * 0.25 );
		c[j] = put( NAN, NAN );
	}
	const scalar_t one = put( 1, 0 ), zero = put( 0, 0 );
	GEMM( a, b, c );
	// Check rows and columns at the start, middle and end outside timing.
	const int indices[] = { 0, n/2, n-1 };
	for ( int row = 0; row < 3; ++row )
	for ( int col = 0; col < 3; ++col )
	{
		const int i = indices[row], j = indices[col];
		value_t sum = { 0, 0 };
		for ( int p = 0; p < n; ++p )
		{
			const value_t x = get( a[i + p*n] ), y = get( b[p + j*n] );
			sum.r += x.r*y.r - x.i*y.i;
			sum.i += x.r*y.i + x.i*y.r;
		}
		const value_t got = get( c[i + j*n] );
		if ( !isfinite( got.r ) || !isfinite( got.i ) ||
		     got.r != sum.r || got.i != sum.i ) return 1;
	}
	// Dyadic inputs keep the reference sums exact at these sizes.
	fprintf( stderr, "check=pass\n" );
	double start = emscripten_get_now();
	do { GEMM( a, b, c ); } while ( emscripten_get_now() - start < 1000 );
	if ( reps == 0 )
	{
		reps = 1;
		double elapsed;
		do
		{
			start = emscripten_get_now();
			for ( int r = 0; r < reps; ++r ) GEMM( a, b, c );
			elapsed = emscripten_get_now() - start;
			if ( elapsed < 30 ) reps *= 2;
		} while ( elapsed < 30 );
		printf( "cal,%.9f\n", elapsed / reps );
	}
	else for ( int sample = 0; sample < 15; ++sample )
	{
		start = emscripten_get_now();
		for ( int r = 0; r < reps; ++r ) GEMM( a, b, c );
		const double elapsed = emscripten_get_now() - start;
		if ( elapsed <= 0 ) return 1;
		printf( "%d,%d,%d,%.9f\n", n, sample, reps, elapsed / reps );
	}
	free( a ); free( b ); free( c );
	bli_finalize();
	return 0;
}
