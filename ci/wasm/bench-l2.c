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

// Direct CBLAS level-2 benchmark for the wasm32 configuration, in double
// precision on square column-major operands of order n. The output protocol
// is that of ci/wasm/bench-gemm.c, so ci/wasm/bench-gemm.py runs the
// comparison between builds. One operation per build:
//
//   (default)            dgemv, no transpose:   y := A x
//   -DBENCH_GEMV_N_ROWX  the same with x a row of an n x n matrix (incx = n),
//                        as in LAPACK's bidiagonal reduction
//   -DBENCH_GEMV_T       dgemv, transpose:      y := A^T x
//   -DBENCH_GEMV_T_ROWX  the same with incx = n
//   -DBENCH_GER          dger:                  A := A +- 2^-20 x y^T
//   -DBENCH_SYMV         dsymv, lower:          y := A x
//   -DBENCH_NRM2         dnrm2 of a vector of length n
//
// Build against the archive under test, for example
//
//   emcc -O2 -msimd128 -DBENCH_GEMV_T -Iinclude/wasm32 ci/wasm/bench-l2.c \
//       lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1 -o bench-dgemv-t.js
//
// Usage: node bench-dgemv-t.js SIZE REPS
//
// REPS = 0 calibrates and prints "cal,<ms per call>". Otherwise the program
// runs a one-second warmup and prints 15 lines "SIZE,sample,REPS,<ms per
// call>". stderr says whether the kernel that the operation reaches is the
// reference kernel or a registered one, and confirms the check.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten/emscripten.h>
#include "blis.h"
#include "cblas.h"

#if   defined( BENCH_GEMV_N_ROWX )
#define BENCH_KER BLIS_AXPYF_KER
#define BENCH_INCX n
#define OP() cblas_dgemv( CblasColMajor, CblasNoTrans, n, n, 1.0, a, n, x, BENCH_INCX, 0.0, y, 1 )
#elif defined( BENCH_GEMV_T )
#define BENCH_KER BLIS_DOTXF_KER
#define BENCH_INCX 1
#define OP() cblas_dgemv( CblasColMajor, CblasTrans, n, n, 1.0, a, n, x, BENCH_INCX, 0.0, y, 1 )
#elif defined( BENCH_GEMV_T_ROWX )
#define BENCH_KER BLIS_DOTXF_KER
#define BENCH_INCX n
#define OP() cblas_dgemv( CblasColMajor, CblasTrans, n, n, 1.0, a, n, x, BENCH_INCX, 0.0, y, 1 )
#elif defined( BENCH_GER )
#define BENCH_KER BLIS_AXPYV_KER
#define BENCH_INCX 1
#define OP() ( cblas_dger( CblasColMajor, n, n, sign*0x1p-20, x, 1, y, 1, a, n ), sign = -sign )
#elif defined( BENCH_SYMV )
#define BENCH_KER BLIS_DOTXAXPYF_KER
#define BENCH_INCX 1
#define OP() cblas_dsymv( CblasColMajor, CblasLower, n, 1.0, a, n, x, BENCH_INCX, 0.0, y, 1 )
#elif defined( BENCH_NRM2 )
#define BENCH_KER BLIS_DOTV_KER
#define BENCH_INCX 1
#define OP() ( nrm = cblas_dnrm2( n, x, BENCH_INCX ) )
#else
#define BENCH_KER BLIS_AXPYF_KER
#define BENCH_INCX 1
#define OP() cblas_dgemv( CblasColMajor, CblasNoTrans, n, n, 1.0, a, n, x, BENCH_INCX, 0.0, y, 1 )
#endif

int main( int argc, char** argv )
{
	if ( argc != 3 )
	{
		fprintf( stderr, "usage: bench-l2.js SIZE REPS (0 to calibrate)\n" );
		return 2;
	}
	const int n = atoi( argv[1] );
	int reps = atoi( argv[2] );
	if ( n < 2 || n > 4096 || reps < 0 ) return 2;
	bli_init();

	// Report whether the timed calls reach a registered kernel.
	cntx_t ref;
	bli_gks_init_ref_cntx( &ref );
	const cntx_t* cntx = bli_gks_query_cntx();
	const int is_ref = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BENCH_KER, cntx ) ==
	                   bli_cntx_get_ukr_dt( BLIS_DOUBLE, BENCH_KER, &ref );
	fprintf( stderr, "ukr=%s\n", is_ref ? "reference" : "registered" );
	bli_cntx_free( &ref );

	// x has room for a row of an n x n matrix.
	const size_t count = ( size_t )n * n;
	double* a = malloc( count * sizeof( double ) );
	double* x = malloc( count * sizeof( double ) );
	double* y = malloc( ( size_t )n * sizeof( double ) );
	if ( !a || !x || !y ) return 1;
	for ( size_t j = 0; j < count; ++j )
	{
		a[j] = ( ( int )( j*13 % 17 ) - 8 ) * 0.25;
		x[j] = ( ( int )( j*11 % 19 ) - 9 ) * 0.25;
	}
	for ( int i = 0; i < n; ++i ) y[i] = ( ( int )( i*7 % 23 ) - 11 ) * 0.25;
	double sign = 1.0, nrm = 0.0;
	( void )sign; ( void )nrm;

	// Dyadic inputs keep these sums exact, so the check is exact except for
	// the norm. It samples elements at the start, middle and end.
	double* a0 = malloc( count * sizeof( double ) );
	double* y0 = malloc( ( size_t )n * sizeof( double ) );
	if ( !a0 || !y0 ) return 1;
	memcpy( a0, a, count * sizeof( double ) );
	memcpy( y0, y, ( size_t )n * sizeof( double ) );
	OP();
	const int idx[] = { 0, n/2, n-1 };
	for ( int r = 0; r < 3; ++r )
	{
		const int i = idx[r];
#if defined( BENCH_GER )
		for ( int s = 0; s < 3; ++s )
		{
			const int j = idx[s];
			if ( a[i + j*n] != a0[i + j*n] + 0x1p-20*x[i]*y0[j] ) return 1;
		}
#elif defined( BENCH_NRM2 )
		( void )i;
		double sum = 0;
		for ( int p = 0; p < n; ++p ) sum += x[p]*x[p];
		if ( !( fabs( nrm - sqrt( sum ) ) <= 1e-13*sqrt( sum ) ) ) return 1;
#else
		double sum = 0;
		for ( int p = 0; p < n; ++p )
		{
#if defined( BENCH_GEMV_T ) || defined( BENCH_GEMV_T_ROWX )
			const double aip = a[p + i*n];
#elif defined( BENCH_SYMV )
			const double aip = p >= i ? a[p + i*n] : a[i + p*n];
#else
			const double aip = a[i + p*n];
#endif
			sum += aip*x[p*BENCH_INCX];
		}
		if ( y[i] != sum ) return 1;
#endif
	}
	fprintf( stderr, "check=pass\n" );
	free( a0 ); free( y0 );

	double start = emscripten_get_now();
	do { OP(); } while ( emscripten_get_now() - start < 1000 );
	if ( reps == 0 )
	{
		reps = 1;
		double elapsed;
		for ( ;; )
		{
			start = emscripten_get_now();
			for ( int r = 0; r < reps; ++r ) OP();
			elapsed = emscripten_get_now() - start;
			if ( elapsed >= 30 ) break;
			reps *= 2;
		}
		printf( "cal,%.9f\n", elapsed / reps );
	}
	else for ( int sample = 0; sample < 15; ++sample )
	{
		start = emscripten_get_now();
		for ( int r = 0; r < reps; ++r ) OP();
		const double elapsed = emscripten_get_now() - start;
		if ( elapsed <= 0 ) return 1;
		printf( "%d,%d,%d,%.9f\n", n, sample, reps, elapsed / reps );
	}
	free( a ); free( x ); free( y );
	bli_finalize();
	return 0;
}
