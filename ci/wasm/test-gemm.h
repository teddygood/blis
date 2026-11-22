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


#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "blis.h"
#include "cblas.h"

static unsigned checks, failures, calls;
static gemm_ukr_ft saved;

static long double sample( size_t i )
{
	return ( ( int )( i*13 % 17 ) - 8 ) * 0.25L;
}

static void check( int ok, const char* what )
{
	++checks;
	if ( !ok )
	{
		if ( failures < 10 ) fprintf( stderr, "%s\n", what );
		++failures;
	}
}

static void compare( long double got, long double want )
{
	// Real inputs are multiples of 1/4 in [-2, 2] and the scalars are
	// dyadic too, so every product and partial sum is exact in the test
	// type at these sizes and the result must match the oracle exactly.
	check( isfinite( got ) && got == want, "scalar oracle mismatch" );
}

static void counted
     (
       dim_t m, dim_t n, dim_t k,
       const void* alpha, const void* a, const void* b,
       const void* beta, void* c, inc_t rs, inc_t cs,
       const auxinfo_t* aux, const cntx_t* ctx
     )
{
	++calls;
	saved( m, n, k, alpha, a, b, beta, c, rs, cs, aux, ctx );
}

static void direct_case( const cntx_t* ctx, dim_t mr, dim_t nr,
                         dim_t m, dim_t n, dim_t k,
                         inc_t rs, inc_t cs, int skew, int scalar_case )
{
	const size_t na = ( k ? ( size_t )mr*k : 1 ) * sizeof( test_type );
	const size_t nb = ( k ? ( size_t )nr*k : 1 ) * sizeof( test_type );
	const size_t offset = skew ? sizeof( test_type ) : 0;
	unsigned char* abase = calloc( na + offset, 1 );
	unsigned char* bbase = calloc( nb + offset, 1 );
	unsigned char* acopy = malloc( na );
	unsigned char* bcopy = malloc( nb );
	test_type* a = ( test_type* )( abase + offset );
	test_type* b = ( test_type* )( bbase + offset );
	for ( dim_t p = 0; p < k; ++p )
	{
		for ( dim_t i = 0; i < mr; ++i )
			a[ p*mr + i ] = i < m ? ( test_type )sample( p*mr + i ) : 0;
		for ( dim_t j = 0; j < nr; ++j )
			b[ p*nr + j ] = j < n ? ( test_type )sample( p*nr + j + 5 ) : 0;
	}
	memcpy( acopy, a, na ); memcpy( bcopy, b, nb );

	inc_t lo = 0, hi = 0;
	for ( dim_t i = 0; i < m; ++i )
	for ( dim_t j = 0; j < n; ++j )
	{
		inc_t off = i*rs + j*cs;
		if ( off < lo ) lo = off;
		if ( off > hi ) hi = off;
	}
	const size_t span = ( size_t )( hi - lo + 1 );
	const size_t bytes = span * sizeof( test_type );
	unsigned char* base = malloc( bytes + offset + 64 );
	unsigned char* mask = calloc( bytes + offset + 64, 1 );
	memset( base, 0xa5, bytes + offset + 64 );
	test_type* c = ( test_type* )( base + 32 + offset ) - lo;
	const long double av[] = { 1.0L, 0.0L, -1.25L };
	const long double bv[] = { 0.0L, -0.0L, 1.0L, -0.5L };
	const long double alpha = av[scalar_case % 3], beta = bv[scalar_case % 4];
	test_type aa = ( test_type )alpha, bb = ( test_type )beta;
	for ( dim_t i = 0; i < m; ++i )
	for ( dim_t j = 0; j < n; ++j )
	{
		test_type* dest = c + i*rs + j*cs;
		*dest = beta == 0.0L ? ( test_type )NAN : ( test_type )sample( i*n + j + 9 );
		memset( mask + ( ( unsigned char* )dest - base ), 1, sizeof( test_type ) );
	}
	auxinfo_t aux = { 0 };
	saved( m, n, k, &aa, a, b, &bb, c, rs, cs, &aux, ctx );
	for ( dim_t i = 0; i < m; ++i )
	for ( dim_t j = 0; j < n; ++j )
	{
		long double sum = 0;
		for ( dim_t p = 0; p < k; ++p )
			sum += ( long double )a[ p*mr + i ] * ( long double )b[ p*nr + j ];
		long double want = alpha * sum;
		if ( beta != 0.0L )
			want += beta * sample( i*n + j + 9 );
		compare( ( long double )c[ i*rs + j*cs ], want );
	}
	for ( size_t s = 0; s < bytes + offset + 64; ++s )
		if ( !mask[s] ) check( base[s] == 0xa5, "output guard overwritten" );
	check( memcmp( a, acopy, na ) == 0 && memcmp( b, bcopy, nb ) == 0,
	       "packed input modified" );
	check( ( long double )aa == alpha && ( long double )bb == beta, "scalar inputs modified" );
	free( abase ); free( bbase ); free( acopy ); free( bcopy );
	free( base ); free( mask );
}

static void public_cases( cntx_t* ctx )
{
	const int trans[] = { CblasNoTrans, CblasTrans, CblasConjTrans };
	// The last shape has m above MC and k above KC for every datatype, so
	// it crosses both cache-blocking boundaries; it skips the conjugate
	// transposes to bound the cost of the oracle.
	const dim_t shapes[][3] = { { 1, 1, 1 }, { 3, 5, 7 }, { 17, 19, 23 },
	                            { 65, 67, 33 }, { 259, 37, 300 } };
	const int nshapes = sizeof( shapes ) / sizeof( shapes[0] );
	for ( int order = 0; order < 2; ++order )
	for ( int ta = 0; ta < 3; ++ta )
	for ( int tb = 0; tb < 3; ++tb )
	for ( int sh = 0; sh < nshapes; ++sh )
	{
		const dim_t m = shapes[sh][0], n = shapes[sh][1], k = shapes[sh][2];
		if ( sh == nshapes - 1 && ( ta == 2 || tb == 2 ) ) continue;
		const dim_t ar = ta ? k : m, ac = ta ? m : k;
		const dim_t br = tb ? n : k, bc = tb ? k : n;
		const dim_t lda = ( order ? ar : ac ) + 3;
		const dim_t ldb = ( order ? br : bc ) + 3;
		const dim_t ldc = ( order ? m : n ) + 3;
		const size_t na = ( size_t )lda * ( size_t )( order ? ac : ar );
		const size_t nb = ( size_t )ldb * ( size_t )( order ? bc : br );
		const size_t nc = ( size_t )ldc * ( size_t )( order ? n : m );
		test_type* a = malloc( na * sizeof( test_type ) );
		test_type* b = malloc( nb * sizeof( test_type ) );
		test_type* c = malloc( nc * sizeof( test_type ) );
		for ( size_t s = 0; s < na; ++s ) a[s] = ( test_type )sample( s );
		for ( size_t s = 0; s < nb; ++s ) b[s] = ( test_type )sample( s + 3 );
		for ( size_t s = 0; s < nc; ++s ) c[s] = ( test_type )sample( s + 7 );
		const long double av = 1.25L, bv = -0.5L;
		const test_type alpha = ( test_type )av, beta = ( test_type )bv;
		calls = 0;
		TEST_CALL( order ? CblasColMajor : CblasRowMajor, trans[ta], trans[tb],
		           m, n, k, alpha, a, lda, b, ldb, beta, c, ldc );
		if ( m > 1 && n > 1 ) check( calls > 0, "CBLAS did not reach SIMD kernel" );
		for ( dim_t i = 0; i < m; ++i )
		for ( dim_t j = 0; j < n; ++j )
		{
			long double sum = 0;
			for ( dim_t p = 0; p < k; ++p )
			{
				const dim_t ai = ta ? p : i, aj = ta ? i : p;
				const dim_t bi = tb ? j : p, bj = tb ? p : j;
				const long double x = a[ order ? ai + aj*lda : ai*lda + aj ];
				const long double y = b[ order ? bi + bj*ldb : bi*ldb + bj ];
				sum += x * y;
			}
			const size_t off = ( size_t )( order ? i + j*ldc : i*ldc + j );
			const long double want = av * sum + bv * sample( off + 7 );
			compare( ( long double )c[ off ], want );
		}
		for ( dim_t i = 0; i < ( order ? n : m ); ++i )
		for ( dim_t j = ( order ? m : n ); j < ldc; ++j )
			check( ( long double )c[ i*ldc + j ] == sample( i*ldc + j + 7 ),
			       "CBLAS padding overwritten" );
		free( a ); free( b ); free( c );
	}
	( void )ctx;
}

int main( void )
{
	bli_init();
	cntx_t* ctx = ( cntx_t* )bli_gks_query_cntx();
	saved = ( gemm_ukr_ft )bli_cntx_get_ukr_dt( TEST_DT, BLIS_GEMM_UKR, ctx );
	const dim_t mr = bli_cntx_get_blksz_def_dt( TEST_DT, BLIS_MR, ctx );
	const dim_t nr = bli_cntx_get_blksz_def_dt( TEST_DT, BLIS_NR, ctx );
#ifdef BLIS_KERNELS_WASM32
	check( saved == ( gemm_ukr_ft )TEST_KERNEL, "wrong registered kernel" );
	check( bli_cntx_get_ukr_prefs_dt( TEST_DT, BLIS_GEMM_UKR_ROW_PREF, ctx ),
	       "wrong row preference" );
	check( mr == TEST_MR &&
	       bli_cntx_get_blksz_max_dt( TEST_DT, BLIS_MR, ctx ) == TEST_MR &&
	       nr == TEST_NR &&
	       bli_cntx_get_blksz_max_dt( TEST_DT, BLIS_NR, ctx ) == TEST_NR,
	       "wrong packing sizes" );
#endif
	const dim_t ks[] = { 0, 1, 3, 4, 7, 31, 255, 256, 257 };
	const inc_t layouts[][2] = { { nr + 3, 1 }, { 1, mr + 3 },
	                            { 2*nr + 3, 2 }, { -( nr + 3 ), 1 },
	                            { 1, -( mr + 3 ) } };
	for ( dim_t m = 0; m <= mr; ++m )
	for ( dim_t n = 0; n <= nr; ++n )
	for ( size_t k = 0; k < sizeof( ks )/sizeof( ks[0] ); ++k )
	for ( int l = 0; l < 5; ++l )
	for ( int s = 0; s < 12; ++s )
		direct_case( ctx, mr, nr, m, n, ks[k], layouts[l][0], layouts[l][1], s % 2, s );

	bli_cntx_set_ukr_dt( ( void_fp )counted, TEST_DT, BLIS_GEMM_UKR, ctx );
	public_cases( ctx );
#ifdef TEST_EXTRA
	TEST_EXTRA( ctx );
#endif
	bli_cntx_set_ukr_dt( ( void_fp )saved, TEST_DT, BLIS_GEMM_UKR, ctx );
	printf( TEST_NAME ": %u checks, %u failures\n", checks, failures );
	bli_finalize();
	return failures ? 1 : 0;
}
