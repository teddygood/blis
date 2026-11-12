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

#include "blis.h"

typedef float test_type;
typedef float test_real;
#define TEST_DT BLIS_FLOAT
#define TEST_COMPLEX 0
#define TEST_MR 4
#define TEST_NR 8
#define TEST_EPS FLT_EPSILON
#define TEST_KERNEL bli_sgemm_wasm32_simd128_4x8
#define TEST_NAME "test-sgemm"
#define TEST_CALL cblas_sgemm
#define TEST_EXTRA test_cgemm_1m
static void test_cgemm_1m( cntx_t* ctx );
#include "test-gemm.h"

// With this kernel registered and the single-complex kernel still the
// reference one, BLIS makes 1m the default method for single-complex
// level-3 operations, so cgemm must run on this kernel. main() has the
// counting wrapper installed when it calls this.
static void test_cgemm_1m( cntx_t* ctx )
{
	( void )ctx;
	check( bli_ind_oper_find_avail( BLIS_GEMM, BLIS_SCOMPLEX ) == BLIS_1M,
	       "single-complex gemm does not default to 1m" );

	enum { M = 17, N = 19, K = 23 };
	static scomplex a[M*K], b[K*N], c[M*N], c0[M*N];
	for ( size_t s = 0; s < M*K; ++s )
		a[s] = ( scomplex ){ ( ( int )( s*13 % 17 ) - 8 ) * 0.25f,
		                     ( ( int )( s*7 % 13 ) - 6 ) * 0.25f };
	for ( size_t s = 0; s < K*N; ++s )
		b[s] = ( scomplex ){ ( ( int )( s*11 % 19 ) - 9 ) * 0.25f,
		                     ( ( int )( s*5 % 11 ) - 5 ) * 0.25f };
	for ( size_t s = 0; s < M*N; ++s )
		c[s] = c0[s] = ( scomplex ){ ( ( int )( s*3 % 7 ) - 3 ) * 0.25f, 0.5f };
	const scomplex alpha = { 1.25f, 0.5f }, beta = { -0.5f, 0.25f };

	calls = 0;
	cblas_cgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
	             &alpha, a, M, b, K, &beta, c, M );
	check( calls > 0, "default cgemm did not reach the SIMD SGEMM kernel" );

	// The data are dyadic, so the exact double-precision result must be
	// reproduced exactly.
	for ( int i = 0; i < M; ++i )
	for ( int j = 0; j < N; ++j )
	{
		double sr = 0, si = 0;
		for ( int p = 0; p < K; ++p )
		{
			const scomplex x = a[i + p*M], y = b[p + j*K];
			sr += ( double )x.real*y.real - ( double )x.imag*y.imag;
			si += ( double )x.real*y.imag + ( double )x.imag*y.real;
		}
		const scomplex z = c0[i + j*M];
		const double wr = alpha.real*sr - alpha.imag*si
		                + beta.real*z.real - beta.imag*z.imag;
		const double wi = alpha.real*si + alpha.imag*sr
		                + beta.real*z.imag + beta.imag*z.real;
		check( c[i + j*M].real == wr && c[i + j*M].imag == wi,
		       "default cgemm result mismatch" );
	}
}
