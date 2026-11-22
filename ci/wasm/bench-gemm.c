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

// Direct CBLAS GEMM benchmark for the wasm32 configuration: square,
// column-major, no transpose, alpha = 1 and beta = 0. Packing is timed;
// allocation and the result check are not. ci/wasm/bench-gemm.py runs the
// comparison between builds.
//
// Build against the archive under test, for example
//
//   emcc -O2 -msimd128 -Iinclude/wasm32 ci/wasm/bench-gemm.c \
//       lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1 -o bench-dgemm.js
//
// DGEMM is the default; add -DBENCH_SGEMM for SGEMM.
//
// Usage: node bench-dgemm.js SIZE REPS
//
// REPS = 0 calibrates and prints "cal,<ms per call>". Otherwise the program
// runs a one-second warmup and prints 15 lines "SIZE,sample,REPS,<ms per
// call>". stderr names the registered microkernel and confirms the check.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten/emscripten.h>
#include "blis.h"
#include "cblas.h"

#if defined( BENCH_SGEMM )
typedef float scalar_t;
#define BENCH_DT   BLIS_FLOAT
#define BENCH_NAME "sgemm"
#define GEMM( a, b, c ) \
	cblas_sgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, \
	             n, n, n, 1.0f, a, n, b, n, 0.0f, c, n )
#else
typedef double scalar_t;
#define BENCH_DT   BLIS_DOUBLE
#define BENCH_NAME "dgemm"
#define GEMM( a, b, c ) \
	cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, \
	             n, n, n, 1.0, a, n, b, n, 0.0, c, n )
#endif

int main( int argc, char** argv )
{
	if ( argc != 3 )
	{
		fprintf( stderr, "usage: bench-%s.js SIZE REPS (0 to calibrate)\n", BENCH_NAME );
		return 2;
	}
	const int n = atoi( argv[1] );
	int reps = atoi( argv[2] );
	if ( n < 2 || n > 4096 || reps < 0 ) return 2;
	bli_init();

	// Report which microkernel the timed calls use.
	cntx_t ref;
	bli_gks_init_ref_cntx( &ref );
	const cntx_t* cntx = bli_gks_query_cntx();
	const int is_ref = bli_cntx_get_ukr_dt( BENCH_DT, BLIS_GEMM_UKR, cntx ) ==
	                   bli_cntx_get_ukr_dt( BENCH_DT, BLIS_GEMM_UKR, &ref );
	fprintf( stderr, "ukr=%s %dx%d\n", is_ref ? "reference" : "registered",
	         ( int )bli_cntx_get_blksz_def_dt( BENCH_DT, BLIS_MR, cntx ),
	         ( int )bli_cntx_get_blksz_def_dt( BENCH_DT, BLIS_NR, cntx ) );
	bli_cntx_free( &ref );

	const size_t count = ( size_t )n * n;
	scalar_t* a = malloc( count * sizeof( scalar_t ) );
	scalar_t* b = malloc( count * sizeof( scalar_t ) );
	scalar_t* c = malloc( count * sizeof( scalar_t ) );
	if ( !a || !b || !c ) return 1;
	for ( size_t j = 0; j < count; ++j )
	{
		a[j] = ( ( int )( j*13 % 17 ) - 8 ) * 0.25;
		b[j] = ( ( int )( j*11 % 19 ) - 9 ) * 0.25;
		c[j] = NAN;
	}
	GEMM( a, b, c );

	// Dyadic inputs keep these sums exact, so the check is exact. It
	// samples rows and columns at the start, middle and end.
	const int idx[] = { 0, n/2, n-1 };
	for ( int r = 0; r < 3; ++r )
	for ( int s = 0; s < 3; ++s )
	{
		const int i = idx[r], j = idx[s];
		double sum = 0;
		for ( int p = 0; p < n; ++p ) sum += ( double )a[i + p*n] * b[p + j*n];
		if ( !isfinite( c[i + j*n] ) || c[i + j*n] != sum ) return 1;
	}
	fprintf( stderr, "check=pass\n" );

	double start = emscripten_get_now();
	do { GEMM( a, b, c ); } while ( emscripten_get_now() - start < 1000 );
	if ( reps == 0 )
	{
		reps = 1;
		double elapsed;
		for ( ;; )
		{
			start = emscripten_get_now();
			for ( int r = 0; r < reps; ++r ) GEMM( a, b, c );
			elapsed = emscripten_get_now() - start;
			if ( elapsed >= 30 ) break;
			reps *= 2;
		}
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
