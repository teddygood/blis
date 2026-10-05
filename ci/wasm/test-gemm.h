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

typedef struct { long double r, i; } value_t;
static unsigned checks, failures, calls;
static gemm_ukr_ft saved;

static value_t add( value_t a, value_t b )
{ return ( value_t ){ a.r + b.r, a.i + b.i }; }

static value_t mul( value_t a, value_t b )
{ return ( value_t ){ a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r }; }

static value_t get( const test_type* p )
{
#if TEST_COMPLEX
	return ( value_t ){ p->real, p->imag };
#else
	return ( value_t ){ *p, 0 };
#endif
}

static void put( test_type* p, value_t v )
{
#if TEST_COMPLEX
	p->real = v.r; p->imag = v.i;
#else
	*p = v.r;
#endif
}

static value_t sample( size_t i )
{
	return ( value_t ){ ( ( int )( i*13 % 17 ) - 8 ) * 0.25L,
	                   TEST_COMPLEX * ( ( int )( i*7 % 13 ) - 6 ) * 0.25L };
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

static void compare( value_t got, value_t want, long double scale, dim_t k )
{
	const long double bound = 64 * TEST_EPS * ( k + 2 ) * ( 1 + scale );
	check( isfinite( got.r ) && isfinite( got.i ) &&
	       fabsl( got.r - want.r ) <= bound &&
	       fabsl( got.i - want.i ) <= bound, "scalar oracle mismatch" );
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

static void direct_case( const cntx_t* ctx, dim_t m, dim_t n, dim_t k,
                         inc_t rs, inc_t cs, int skew, int scalar_case )
{
	const dim_t mr = TEST_MR, nr = TEST_NR;
	const size_t na = ( k ? mr*k : 1 ) * sizeof( test_type );
	const size_t nb = ( k ? nr*k : 1 ) * sizeof( test_type );
	const size_t offset = skew ? sizeof( test_real ) : 0;
	unsigned char* abase = malloc( na + offset );
	unsigned char* bbase = malloc( nb + offset );
	unsigned char* acopy = malloc( na );
	unsigned char* bcopy = malloc( nb );
	test_type* a = ( test_type* )( abase + offset );
	test_type* b = ( test_type* )( bbase + offset );
	for ( dim_t p = 0; p < k; ++p )
	{
		for ( dim_t i = 0; i < mr; ++i )
			put( a + p*mr + i, i < m ? sample( p*mr + i ) : ( value_t ){ 0, 0 } );
		for ( dim_t j = 0; j < nr; ++j )
			put( b + p*nr + j, j < n ? sample( p*nr + j + 5 ) : ( value_t ){ 0, 0 } );
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
	const size_t span = hi - lo + 1;
	const size_t bytes = span * sizeof( test_type );
	unsigned char* base = malloc( bytes + offset + 64 );
	unsigned char* mask = calloc( bytes + offset + 64, 1 );
	memset( base, 0xa5, bytes + offset + 64 );
	test_type* c = ( test_type* )( base + 32 + offset ) - lo;
	const value_t av[] = { { 1, 0 }, { 0, 0 }, { -1.25L, TEST_COMPLEX*0.5L } };
	const value_t bv[] = { { 0, 0 }, { -0.0L, 0 }, { 1, 0 }, { -0.5L, TEST_COMPLEX*0.25L } };
	const value_t alpha = av[scalar_case % 3], beta = bv[scalar_case % 4];
	test_type aa, bb;
	put( &aa, alpha ); put( &bb, beta );
	for ( dim_t i = 0; i < m; ++i )
	for ( dim_t j = 0; j < n; ++j )
	{
		test_type* dest = c + i*rs + j*cs;
		put( dest, beta.r == 0 && beta.i == 0 ? ( value_t ){ NAN, TEST_COMPLEX ? NAN : 0 } : sample( i*n + j + 9 ) );
		memset( mask + ( ( unsigned char* )dest - base ), 1, sizeof( test_type ) );
	}
	auxinfo_t aux = { 0 };
	saved( m, n, k, &aa, a, b, &bb, c, rs, cs, &aux, ctx );
	for ( dim_t i = 0; i < m; ++i )
	for ( dim_t j = 0; j < n; ++j )
	{
		value_t sum = { 0, 0 };
		long double scale = 0;
		for ( dim_t p = 0; p < k; ++p )
		{
			value_t v = mul( get( a + p*mr + i ), get( b + p*nr + j ) );
			sum = add( sum, v ); scale += fabsl( v.r ) + fabsl( v.i );
		}
		value_t want = mul( alpha, sum );
		if ( beta.r != 0 || beta.i != 0 )
			want = add( want, mul( beta, sample( i*n + j + 9 ) ) );
		compare( get( c + i*rs + j*cs ), want, 4*scale + 16, k );
	}
	for ( size_t s = 0; s < bytes + offset + 64; ++s )
		if ( !mask[s] ) check( base[s] == 0xa5, "output guard overwritten" );
	check( memcmp( a, acopy, na ) == 0 && memcmp( b, bcopy, nb ) == 0,
	       "packed input modified" );
	check( get( &aa ).r == alpha.r && get( &aa ).i == alpha.i &&
	       get( &bb ).r == beta.r && get( &bb ).i == beta.i, "scalar inputs modified" );
	free( abase ); free( bbase ); free( acopy ); free( bcopy );
	free( base ); free( mask );
}

static void public_cases( cntx_t* ctx, int native )
{
	const int trans[] = { CblasNoTrans, CblasTrans, CblasConjTrans };
	const dim_t shapes[][3] = { { 1, 1, 1 }, { 3, 5, 7 }, { 17, 19, 23 }, { 65, 67, 33 } };
	for ( int order = 0; order < 2; ++order )
	for ( int ta = 0; ta < 3; ++ta )
	for ( int tb = 0; tb < 3; ++tb )
	for ( int sh = 0; sh < 4; ++sh )
	{
		const dim_t m = shapes[sh][0], n = shapes[sh][1], k = shapes[sh][2];
		const dim_t ar = ta ? k : m, ac = ta ? m : k;
		const dim_t br = tb ? n : k, bc = tb ? k : n;
		const dim_t lda = ( order ? ar : ac ) + 3;
		const dim_t ldb = ( order ? br : bc ) + 3;
		const dim_t ldc = ( order ? m : n ) + 3;
		const size_t na = lda * ( order ? ac : ar );
		const size_t nb = ldb * ( order ? bc : br );
		const size_t nc = ldc * ( order ? n : m );
		test_type* a = malloc( na * sizeof( test_type ) );
		test_type* b = malloc( nb * sizeof( test_type ) );
		test_type* c = malloc( nc * sizeof( test_type ) );
		for ( size_t s = 0; s < na; ++s ) put( a + s, sample( s ) );
		for ( size_t s = 0; s < nb; ++s ) put( b + s, sample( s + 3 ) );
		for ( size_t s = 0; s < nc; ++s ) put( c + s, sample( s + 7 ) );
		test_type alpha, beta;
		const value_t av = { 1.25L, TEST_COMPLEX*0.5L };
		const value_t bv = { -0.5L, TEST_COMPLEX*0.25L };
		put( &alpha, av ); put( &beta, bv );
		calls = 0;
		TEST_CALL( order ? CblasColMajor : CblasRowMajor, trans[ta], trans[tb],
		           m, n, k, alpha, a, lda, b, ldb, beta, c, ldc );
		if ( native && m > 1 && n > 1 ) check( calls > 0, "CBLAS did not reach SIMD kernel" );
		for ( dim_t i = 0; i < m; ++i )
		for ( dim_t j = 0; j < n; ++j )
		{
			value_t sum = { 0, 0 };
			long double scale = 0;
			for ( dim_t p = 0; p < k; ++p )
			{
				const dim_t ai = ta ? p : i, aj = ta ? i : p;
				const dim_t bi = tb ? j : p, bj = tb ? p : j;
				value_t x = get( a + ( order ? ai + aj*lda : ai*lda + aj ) );
				value_t y = get( b + ( order ? bi + bj*ldb : bi*ldb + bj ) );
				if ( ta == 2 ) x.i = -x.i;
				if ( tb == 2 ) y.i = -y.i;
				value_t v = mul( x, y );
				sum = add( sum, v ); scale += fabsl( v.r ) + fabsl( v.i );
			}
			const size_t off = order ? i + j*ldc : i*ldc + j;
			value_t want = add( mul( av, sum ), mul( bv, sample( off + 7 ) ) );
			compare( get( c + off ), want, 4*scale + 16, k );
		}
		for ( dim_t i = 0; i < ( order ? n : m ); ++i )
		for ( dim_t j = ( order ? m : n ); j < ldc; ++j )
			check( get( c + i*ldc + j ).r == sample( i*ldc + j + 7 ).r &&
			       get( c + i*ldc + j ).i == sample( i*ldc + j + 7 ).i, "CBLAS padding overwritten" );
		free( a ); free( b ); free( c );
	}
	( void )ctx;
}

int main( void )
{
	bli_init();
	cntx_t* ctx = ( cntx_t* )bli_gks_query_cntx();
	saved = ( gemm_ukr_ft )bli_cntx_get_ukr_dt( TEST_DT, BLIS_GEMM_UKR, ctx );
	check( saved == ( gemm_ukr_ft )TEST_KERNEL, "wrong registered kernel" );
	check( bli_cntx_get_blksz_def_dt( TEST_DT, BLIS_MR, ctx ) == TEST_MR &&
	       bli_cntx_get_blksz_max_dt( TEST_DT, BLIS_MR, ctx ) == TEST_MR &&
	       bli_cntx_get_blksz_def_dt( TEST_DT, BLIS_NR, ctx ) == TEST_NR &&
	       bli_cntx_get_blksz_max_dt( TEST_DT, BLIS_NR, ctx ) == TEST_NR, "wrong packing sizes" );
	const dim_t ks[] = { 0, 1, 3, 4, 7, 31, 255, 256, 257 };
	const inc_t layouts[][2] = { { TEST_NR + 3, 1 }, { 1, TEST_MR + 3 },
	                            { 2*TEST_NR + 3, 2 }, { -( TEST_NR + 3 ), 1 },
	                            { 1, -( TEST_MR + 3 ) } };
	for ( dim_t m = 0; m <= TEST_MR; ++m )
	for ( dim_t n = 0; n <= TEST_NR; ++n )
	for ( size_t k = 0; k < sizeof( ks )/sizeof( ks[0] ); ++k )
	for ( int l = 0; l < 5; ++l )
	for ( int s = 0; s < 12; ++s )
		direct_case( ctx, m, n, ks[k], layouts[l][0], layouts[l][1], s % 2, s );

	bli_cntx_set_ukr_dt( ( void_fp )counted, TEST_DT, BLIS_GEMM_UKR, ctx );
#if TEST_COMPLEX
	bli_ind_disable_dt( BLIS_1M, TEST_DT );
#endif
	public_cases( ctx, 1 );
#if TEST_COMPLEX
	bli_ind_enable_dt( BLIS_1M, TEST_DT );
	public_cases( ctx, 0 );
#endif
	bli_cntx_set_ukr_dt( ( void_fp )saved, TEST_DT, BLIS_GEMM_UKR, ctx );
	printf( TEST_NAME ": %u checks, %u failures\n", checks, failures );
	bli_finalize();
	return failures ? 1 : 0;
}
