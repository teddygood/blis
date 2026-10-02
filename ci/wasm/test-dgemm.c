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

// Deterministic contract test for the wasm32 double-precision gemm
// microkernel and the public cblas entry points that dispatch to it.
// Register block sizes are queried from the runtime context instead of
// being hardcoded, so the same source validates the reference and the
// optimized tile. Exits nonzero when any check fails.

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blis.h"
#include "cblas.h"

// -- result accounting ----------------------------------------------------------

static unsigned long n_check = 0;
static unsigned long n_fail  = 0;

static void fail( const char* what )
{
	if ( n_fail < 25 ) fprintf( stderr, "FAIL: %s\n", what );
	++n_fail;
}

#define CHECK( cond, what ) \
	do { ++n_check; if ( !( cond ) ) fail( what ); } while ( 0 )

// -- deterministic values --------------------------------------------------------

static uint32_t rng_state;

static void rng_seed( uint32_t s ) { rng_state = s ? s : 1u; }

static uint32_t rng_next( void )
{
	rng_state = rng_state * 1664525u + 1013904223u;
	return rng_state;
}

// Small signed multiples of 0.25 in [-2, 2]; exactly representable.
static double rng_val( void )
{
	return ( double )( ( int )( ( rng_next() >> 8 ) % 17u ) - 8 ) * 0.25;
}

// Integers in [-4, 4]; sums of small products stay exactly representable.
static double rng_int( void )
{
	return ( double )( ( int )( ( rng_next() >> 8 ) % 9u ) - 4 );
}

// -- sentinels --------------------------------------------------------------------

// Marker bits fill every byte that must stay untouched; both patterns are
// NaNs so an accidental read can never be mistaken for real data.
#define MARK_BITS 0x7ff1a5a5a5a5a5a5ULL
#define CNAN_BITS 0x7ff8deadbee0f00dULL

static void set_bits( double* p, uint64_t u )
{
	memcpy( p, &u, sizeof( u ) );
}

static int bits_eq( const double* p, uint64_t u )
{
	uint64_t v;
	memcpy( &v, p, sizeof( v ) );
	return v == u;
}

// Guarded allocation of n doubles: returns a pointer with GUARD marker
// elements before and after the region, plus an optional 8-byte skew.
#define GUARD 16

static double* galloc( size_t n, int misal, double** base_out )
{
	const size_t tot = n + 2*GUARD + ( misal ? 1 : 0 );
	double*      base = malloc( tot * sizeof( double ) );
	if ( !base ) { fprintf( stderr, "out of memory\n" ); exit( 2 ); }

	for ( size_t i = 0; i < tot; ++i ) set_bits( base + i, MARK_BITS );

	*base_out = base;
	return base + ( misal ? 1 : 0 ) + GUARD;
}

// Slots outside the n-element region must still hold the marker.
static void check_guards( const char* what, const double* p, size_t n, int misal )
{
	const size_t off  = ( misal ? 1 : 0 ) + GUARD;
	const double* base = p - off;
	const size_t  tot  = n + 2*GUARD + ( misal ? 1 : 0 );

	for ( size_t i = 0; i < off; ++i )
		if ( !bits_eq( base + i, MARK_BITS ) ) { fail( what ); return; }
	for ( size_t i = off + n; i < tot; ++i )
		if ( !bits_eq( base + i, MARK_BITS ) ) { fail( what ); return; }
}

// -- direct microkernel calls ------------------------------------------------------

static const dim_t  K_SET[]   = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 31, 255, 256, 257 };
static const dim_t  K_EXACT[] = { 1, 4 };
static const double A_SET[]   = { 0.0, 1.0, -1.0, 1.2 };
static const double B_SET[]   = { 0.0, -0.0, 1.0, -1.0, 0.5 };

typedef struct
{
	inc_t rs, cs;
} lay_t;

typedef struct
{
	gemm_ukr_ft   ukr;
	const cntx_t* cntx;
	dim_t         mr, nr;   // register tile
	dim_t         pmr, pnr; // packed panel strides
} ukr_env_t;

// Call the selected double gemm microkernel once on hand-packed panels and
// verify the C tile elementwise against a scalar oracle while every other
// byte of the C buffer stays at the marker value.
static void ukr_case( const ukr_env_t* e,
                      dim_t m, dim_t n, dim_t k,
                      double alpha_v, double beta_v,
                      inc_t rs, inc_t cs,
                      int c_nan, int misal, int exact )
{
	char what[224];
	snprintf( what, sizeof( what ),
	          "ukr m=%d n=%d k=%d alpha=%g beta=%g rs=%d cs=%d nan=%d misal=%d",
	          ( int )m, ( int )n, ( int )k, alpha_v, beta_v,
	          ( int )rs, ( int )cs, c_nan, misal );

	const dim_t pmr = e->pmr, pnr = e->pnr;

	// Scalar operands get their own exact-size allocations so that a
	// vector load of alpha or beta trips bounds instrumentation.
	double* alpha = malloc( sizeof( double ) );
	double* beta  = malloc( sizeof( double ) );
	double* alog  = malloc( ( (size_t)m * k ? (size_t)m * k : 1 ) * sizeof( double ) );
	double* blog  = malloc( ( (size_t)k * n ? (size_t)k * n : 1 ) * sizeof( double ) );
	double* clog  = malloc( ( (size_t)m * n ? (size_t)m * n : 1 ) * sizeof( double ) );
	// Packed panels: A[i + p*pmr], B[j + p*pnr]; lanes outside m/n are
	// zero padding, matching real packm output. Allocations end exactly
	// at the last panel element so bounds instrumentation sees overruns.
	const size_t na = ( (size_t)pmr * k ? (size_t)pmr * k : 1 ) + ( misal ? 1 : 0 );
	const size_t nb = ( (size_t)pnr * k ? (size_t)pnr * k : 1 ) + ( misal ? 1 : 0 );
	double* a_base = malloc( na * sizeof( double ) );
	double* b_base = malloc( nb * sizeof( double ) );
	double* a_copy = malloc( na * sizeof( double ) );
	double* b_copy = malloc( nb * sizeof( double ) );
	if ( !alpha || !beta || !alog || !blog || !clog ||
	     !a_base || !b_base || !a_copy || !b_copy )
		{ fprintf( stderr, "out of memory\n" ); exit( 2 ); }

	*alpha = alpha_v;
	*beta  = beta_v;

	double* ap = a_base + ( misal ? 1 : 0 );
	double* bp = b_base + ( misal ? 1 : 0 );

	for ( dim_t p = 0; p < k; ++p )
	for ( dim_t i = 0; i < pmr; ++i )
	{
		const double v = i < m ? ( exact ? rng_int() : rng_val() ) : 0.0;
		ap[ i + p*pmr ] = v;
		if ( i < m ) alog[ i*k + p ] = v;
	}
	for ( dim_t p = 0; p < k; ++p )
	for ( dim_t j = 0; j < pnr; ++j )
	{
		const double v = j < n ? ( exact ? rng_int() : rng_val() ) : 0.0;
		bp[ j + p*pnr ] = v;
		if ( j < n ) blog[ p*n + j ] = v;
	}
	memcpy( a_copy, a_base, na * sizeof( double ) );
	memcpy( b_copy, b_base, nb * sizeof( double ) );

	// C storage extent: with negative strides the logical footprint does
	// not start at the base pointer, so shift the origin accordingly.
	inc_t lo = 0, hi = 0;
	for ( dim_t i = 0; i < m; ++i )
	for ( dim_t j = 0; j < n; ++j )
	{
		const inc_t off = i*rs + j*cs;
		if ( off < lo ) lo = off;
		if ( off > hi ) hi = off;
	}
	const size_t span = (size_t)( hi - lo + 1 );

	double*  c_base;
	double*  c0     = galloc( span, misal, &c_base ) - lo;
	uint8_t* lseen  = calloc( span + 1, 1 );
	if ( !lseen ) { fprintf( stderr, "out of memory\n" ); exit( 2 ); }

	rng_seed( ( uint32_t )( m * 131 + n * 17 + k ) + 3u );
	for ( dim_t i = 0; i < m; ++i )
	for ( dim_t j = 0; j < n; ++j )
	{
		const double v = rng_val();
		clog[ i*n + j ] = v;
		if ( c_nan ) set_bits( c0 + i*rs + j*cs, CNAN_BITS );
		else         c0[ i*rs + j*cs ] = v;
		lseen[ i*rs + j*cs - lo ] = 1;
	}

	auxinfo_t aux;
	memset( &aux, 0, sizeof( aux ) );

	e->ukr( m, n, k, alpha, ap, bp, beta, c0, rs, cs, &aux, e->cntx );

	// Panels and scalars are read-only inputs.
	CHECK( memcmp( ap, a_copy + ( misal ? 1 : 0 ), (size_t)pmr * k * sizeof( double ) ) == 0,
	       "ukr modified packed A" );
	CHECK( memcmp( bp, b_copy + ( misal ? 1 : 0 ), (size_t)pnr * k * sizeof( double ) ) == 0,
	       "ukr modified packed B" );
	CHECK( *alpha == alpha_v && *beta == beta_v, "ukr modified scalars" );

	const int beta0 = ( beta_v == 0.0 );

	for ( dim_t i = 0; i < m; ++i )
	for ( dim_t j = 0; j < n; ++j )
	{
		// Scalar oracle accumulating in the same order the kernel does.
		double sum = 0.0, absum = 0.0;
		for ( dim_t p = 0; p < k; ++p )
		{
			const double t = alog[ i*k + p ] * blog[ p*n + j ];
			sum   += t;
			absum += fabs( t );
		}
		sum   *= alpha_v;
		absum *= fabs( alpha_v );

		double expect = sum;
		double bound  = absum;
		if ( !beta0 )
		{
			expect += beta_v * clog[ i*n + j ];
			bound  += fabs( beta_v * clog[ i*n + j ] );
		}
		bound = 16.0 * DBL_EPSILON * ( ( double )k + 2.0 )
		        * ( bound > 1.0 ? bound : 1.0 );

		const double got = c0[ i*rs + j*cs ];
		if ( exact )
		{
			uint64_t eu;
			memcpy( &eu, &expect, sizeof( eu ) );
			if ( !bits_eq( &got, eu ) ) fail( "ukr exact-oracle mismatch" );
		}
		else if ( !isfinite( got ) || !isfinite( got - expect ) ||
		          fabs( got - expect ) > bound )
		{
			fail( what );
		}
	}

	// Every nonlogical slot in the C span must still be the marker.
	for ( size_t s = 0; s < span; ++s )
	{
		if ( lseen[ s ] ) continue;
		if ( !bits_eq( c0 + ( inc_t )s + lo, MARK_BITS ) )
			{ fail( "ukr wrote outside the logical C tile" ); break; }
	}
	check_guards( "ukr overwrote a C guard region", c0 + lo, span, misal );

	free( lseen );
	free( c_base );
	free( alpha ); free( beta );
	free( alog ); free( blog ); free( clog );
	free( a_base ); free( b_base );
	free( a_copy ); free( b_copy );
}

static void test_ukr( const cntx_t* cntx )
{
	ukr_env_t e;
	e.cntx = cntx;
	e.mr   = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MR, cntx );
	e.nr   = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NR, cntx );
	e.pmr  = bli_cntx_get_blksz_max_dt( BLIS_DOUBLE, BLIS_MR, cntx );
	e.pnr  = bli_cntx_get_blksz_max_dt( BLIS_DOUBLE, BLIS_NR, cntx );
	e.ukr  = ( gemm_ukr_ft )bli_func_get_dt
	         ( BLIS_DOUBLE, bli_cntx_get_ukrs( BLIS_GEMM_UKR, cntx ) );

	CHECK( e.ukr != NULL, "no double gemm microkernel in context" );
	CHECK( e.mr > 0 && e.nr > 0 && e.pmr >= e.mr && e.pnr >= e.nr,
	       "invalid double register/packing blocksizes" );
	if ( !e.ukr ) return;

	const lay_t lays[] =
	{
		{ e.nr + 3,        1 },   // row-contiguous with gaps
		{ 1,        e.mr + 3 },   // column-contiguous with gaps
		{ 2*e.nr + 3,      2 },   // general strides
		{ -( e.nr + 3 ),   1 },   // reversed rows
		{ 1, -( e.mr + 3 ) },     // reversed columns
	};
	const int nlays = ( int )( sizeof( lays ) / sizeof( lays[0] ) );

	int parity = 0;
	for ( dim_t m = 1; m <= e.mr; ++m )
	for ( dim_t n = 1; n <= e.nr; ++n )
	for ( size_t ik = 0; ik < sizeof( K_SET )/sizeof( K_SET[0] ); ++ik )
	for ( int il = 0; il < nlays; ++il )
	{
		const dim_t  k  = K_SET[ik];
		const lay_t* ly = &lays[il];

		// The row-contiguous and general layouts cover every alpha; the
		// rest still cover every beta, including both zeros.
		const int n_a = ( il == 0 || il == 2 ) ? 4 : 2;
		for ( int ia = 0; ia < n_a; ++ia )
		for ( size_t ib = 0; ib < sizeof( B_SET )/sizeof( B_SET[0] ); ++ib )
		{
			const double bv = B_SET[ib];
			ukr_case( &e, m, n, k, A_SET[ia], bv, ly->rs, ly->cs,
			          bv == 0.0, parity & 1, 0 );
			++parity;
		}
	}

	// Exactly representable small-integer products must come out exact.
	for ( dim_t m = 1; m <= e.mr; ++m )
	for ( dim_t n = 1; n <= e.nr; ++n )
	for ( size_t ik = 0; ik < sizeof( K_EXACT )/sizeof( K_EXACT[0] ); ++ik )
	for ( int il = 0; il < 2; ++il )
		ukr_case( &e, m, n, K_EXACT[ik], 1.0, 0.0,
		          lays[il].rs, lays[il].cs, 0, ik & 1, 1 );

	// Empty tiles must not write anything.
	for ( int il = 0; il < nlays; ++il )
	{
		ukr_case( &e, 0, e.nr, 7, 1.2, -1.0, lays[il].rs, lays[il].cs, 0, 0, 0 );
		ukr_case( &e, e.mr, 0, 7, 1.2, -1.0, lays[il].rs, lays[il].cs, 0, 1, 0 );
		ukr_case( &e, 0, 0,    7, 1.2,  0.0, lays[il].rs, lays[il].cs, 0, 0, 0 );
	}
}

// -- helpers for CBLAS-laid-out matrices -------------------------------------------

// Element (r,c) of a CBLAS-stored matrix (r,c index the stored matrix).
static double mget( const double* p, int order, dim_t ld, dim_t r, dim_t c )
{
	return order == CblasRowMajor ? p[ r*ld + c ] : p[ r + c*ld ];
}

static void mset( double* p, int order, dim_t ld, dim_t r, dim_t c, double v )
{
	if ( order == CblasRowMajor ) p[ r*ld + c ] = v;
	else                          p[ r + c*ld ] = v;
}

static size_t cmat_size( int order, dim_t ld, dim_t rows, dim_t cols )
{
	return (size_t)ld * (size_t)( order == CblasRowMajor ? rows : cols );
}

// Allocate a stored rows x cols matrix with a padded leading dimension;
// slots past the logical dimension inside each leading-dim stride keep
// the marker.
static double* cmat_alloc( int order, dim_t rows, dim_t cols,
                           dim_t* ldp, double** base_out )
{
	const dim_t ld = ( order == CblasRowMajor ? cols : rows ) + 3;
	*ldp = ld;
	return galloc( cmat_size( order, ld, rows, cols ), 0, base_out );
}

// Marker-check the leading-dimension padding of a stored rows x cols
// matrix.
static void cmat_check_edges( const char* what, const double* p, int order,
                              dim_t ld, dim_t rows, dim_t cols )
{
	if ( order == CblasRowMajor )
	{
		for ( dim_t i = 0; i < rows; ++i )
		for ( dim_t s = cols; s < ld; ++s )
			if ( !bits_eq( p + i*ld + s, MARK_BITS ) ) { fail( what ); return; }
	}
	else
	{
		for ( dim_t j = 0; j < cols; ++j )
		for ( dim_t s = rows; s < ld; ++s )
			if ( !bits_eq( p + s + j*ld, MARK_BITS ) ) { fail( what ); return; }
	}
}

// -- public cblas_dgemm -------------------------------------------------------------

static void test_cblas_gemm( void )
{
	static const dim_t SHAPES[][3] =
	{
		{ 1, 1, 1 }, { 3, 3, 3 }, { 4, 4, 4 }, { 5, 5, 5 },
		{ 7, 7, 7 }, { 8, 8, 8 }, { 9, 9, 9 }, { 3, 5, 7 },
		{ 5, 3, 9 }, { 4, 9, 5 },
		{ 127, 127, 127 }, { 128, 128, 128 }, { 129, 129, 129 },
		{ 127, 129, 255 }, { 129, 127, 257 }, { 255, 255, 255 },
		{ 256, 256, 257 }, { 257, 256, 255 },
	};
	static const int ORDERS[] = { CblasRowMajor, CblasColMajor };
	static const int TRANS[]  = { CblasNoTrans, CblasTrans };

	for ( size_t is = 0; is < sizeof( SHAPES )/sizeof( SHAPES[0] ); ++is )
	{
		const dim_t m = SHAPES[is][0], n = SHAPES[is][1], k = SHAPES[is][2];
		const int   small = ( m <= 9 && n <= 9 && k <= 9 );

		for ( int io = 0; io < 2; ++io )
		for ( int ta = 0; ta < 2; ++ta )
		for ( int tb = 0; tb < 2; ++tb )
		{
			// Small shapes cover every combination; large shapes sample
			// half of them to bound the run time.
			if ( !small && ( io + ta + tb ) % 2 ) continue;

			const int order  = ORDERS[io];
			const int transa = TRANS[ta];
			const int transb = TRANS[tb];

			char what[160];
			snprintf( what, sizeof( what ),
			          "gemm m=%d n=%d k=%d ord=%d ta=%d tb=%d",
			          ( int )m, ( int )n, ( int )k, order, transa, transb );

			const dim_t ar = transa == CblasNoTrans ? m : k;
			const dim_t ac = transa == CblasNoTrans ? k : m;
			const dim_t br = transb == CblasNoTrans ? k : n;
			const dim_t bc = transb == CblasNoTrans ? n : k;

			dim_t lda, ldb, ldc;
			double *a_base, *b_base, *c_base;
			double* a = cmat_alloc( order, ar, ac, &lda, &a_base );
			double* b = cmat_alloc( order, br, bc, &ldb, &b_base );
			double* c = cmat_alloc( order, m,  n,  &ldc, &c_base );

			rng_seed( ( uint32_t )( m * 37 + n * 7 + k ) + 11u );
			for ( dim_t i = 0; i < ar; ++i )
			for ( dim_t j = 0; j < ac; ++j ) mset( a, order, lda, i, j, rng_val() );
			for ( dim_t i = 0; i < br; ++i )
			for ( dim_t j = 0; j < bc; ++j ) mset( b, order, ldb, i, j, rng_val() );

			const size_t na = cmat_size( order, lda, ar, ac );
			const size_t nb = cmat_size( order, ldb, br, bc );
			double* a_sv = malloc( na * sizeof( double ) );
			double* b_sv = malloc( nb * sizeof( double ) );
			double* clog = malloc( (size_t)m * n * sizeof( double ) );
			if ( !a_sv || !b_sv || !clog )
				{ fprintf( stderr, "out of memory\n" ); exit( 2 ); }

			for ( dim_t i = 0; i < m; ++i )
			for ( dim_t j = 0; j < n; ++j )
			{
				const double v = rng_val();
				mset( c, order, ldc, i, j, v );
				clog[ i*n + j ] = v;
			}
			memcpy( a_sv, a, na * sizeof( double ) );
			memcpy( b_sv, b, nb * sizeof( double ) );

			// Exercise alpha=0 and beta=0 on a slice of the small shapes.
			const double alpha_v = small && ta == 0 && tb == 1 ? 0.0 : 1.2;
			const double beta_v  = small && ta == 0 ? 0.5 : -1.0;

			cblas_dgemm( order, transa, transb, m, n, k, alpha_v,
			             a, lda, b, ldb, beta_v, c, ldc );

			CHECK( memcmp( a, a_sv, na * sizeof( double ) ) == 0,
			       "gemm modified A" );
			CHECK( memcmp( b, b_sv, nb * sizeof( double ) ) == 0,
			       "gemm modified B" );

			for ( dim_t i = 0; i < m; ++i )
			for ( dim_t j = 0; j < n; ++j )
			{
				double sum = 0.0, absum = 0.0;
				for ( dim_t p = 0; p < k; ++p )
				{
					const double av = transa == CblasNoTrans
					  ? mget( a, order, lda, i, p ) : mget( a, order, lda, p, i );
					const double bv = transb == CblasNoTrans
					  ? mget( b, order, ldb, p, j ) : mget( b, order, ldb, j, p );
					const double t = av * bv;
					sum   += t;
					absum += fabs( t );
				}
				const double expect = beta_v * clog[ i*n + j ] + alpha_v * sum;
				const double bound  = 32.0 * DBL_EPSILON * ( ( double )k + 2.0 )
				  * ( 1.0 + fabs( alpha_v ) * absum + fabs( beta_v * clog[ i*n + j ] ) );
				const double got = mget( c, order, ldc, i, j );
				if ( !isfinite( got ) || fabs( got - expect ) > bound )
					fail( what );
			}

			cmat_check_edges( "gemm wrote outside logical C", c, order, ldc, m, n );
			check_guards( "gemm overwrote a C guard region", c,
			              cmat_size( order, ldc, m, n ), 0 );

			free( a_sv ); free( b_sv ); free( clog );
			free( a_base ); free( b_base ); free( c_base );
		}
	}

	// Degenerate public calls: nothing may be written when m or n is
	// zero, and k=0 must reduce to an exact beta*C update.
	for ( int io = 0; io < 2; ++io )
	{
		const int order = ORDERS[io];
		dim_t lda, ldc;
		double *a_base, *c_base;
		double* a = cmat_alloc( order, 8, 8, &lda, &a_base );
		double* c = cmat_alloc( order, 4, 4, &ldc, &c_base );

		cblas_dgemm( order, CblasNoTrans, CblasNoTrans, 0, 4, 4, 1.2,
		             a, lda, a, lda, -1.0, c, ldc );
		cblas_dgemm( order, CblasNoTrans, CblasNoTrans, 4, 0, 4, 1.2,
		             a, lda, a, lda, -1.0, c, ldc );
		cblas_dgemm( order, CblasNoTrans, CblasNoTrans, 0, 0, 4, 1.2,
		             a, lda, a, lda, -1.0, c, ldc );
		for ( dim_t i = 0; i < 4; ++i )
		for ( dim_t s = 0; s < ldc; ++s )
		{
			const double* p = order == CblasRowMajor ? c + i*ldc + s : c + s + i*ldc;
			if ( !bits_eq( p, MARK_BITS ) ) fail( "empty gemm wrote C" );
		}

		for ( dim_t i = 0; i < 4; ++i )
		for ( dim_t j = 0; j < 4; ++j ) mset( c, order, ldc, i, j, 2.0 );
		cblas_dgemm( order, CblasNoTrans, CblasNoTrans, 4, 4, 0, 1.2,
		             a, lda, a, lda, -0.5, c, ldc );
		for ( dim_t i = 0; i < 4; ++i )
		for ( dim_t j = 0; j < 4; ++j )
			CHECK( mget( c, order, ldc, i, j ) == -1.0, "k=0 gemm wrong" );
		cblas_dgemm( order, CblasNoTrans, CblasNoTrans, 4, 4, 0, 1.2,
		             a, lda, a, lda, 0.0, c, ldc );
		for ( dim_t i = 0; i < 4; ++i )
		for ( dim_t j = 0; j < 4; ++j )
			CHECK( mget( c, order, ldc, i, j ) == 0.0, "k=0 beta=0 gemm wrong" );

		free( a_base ); free( c_base );
	}
}

// -- public cblas_dtrsm ---------------------------------------------------------------

static void test_cblas_trsm( void )
{
	static const dim_t SHAPES[][2] =
		{ { 3, 5 }, { 5, 3 }, { 4, 4 }, { 7, 9 }, { 9, 7 } };
	static const int ORDERS[] = { CblasRowMajor, CblasColMajor };
	static const int SIDES[]  = { CblasLeft, CblasRight };
	static const int UPLOS[]  = { CblasUpper, CblasLower };
	static const int TRANS[]  = { CblasNoTrans, CblasTrans };
	static const int DIAGS[]  = { CblasNonUnit, CblasUnit };

	for ( size_t is = 0; is < sizeof( SHAPES )/sizeof( SHAPES[0] ); ++is )
	for ( int io = 0; io < 2; ++io )
	for ( int sd = 0; sd < 2; ++sd )
	for ( int up = 0; up < 2; ++up )
	for ( int ta = 0; ta < 2; ++ta )
	for ( int dg = 0; dg < 2; ++dg )
	{
		const dim_t m = SHAPES[is][0], n = SHAPES[is][1];
		const int order = ORDERS[io], side = SIDES[sd], uplo = UPLOS[up],
		          trans = TRANS[ta], diag = DIAGS[dg];
		const dim_t dim = side == CblasLeft ? m : n;

		char what[160];
		snprintf( what, sizeof( what ),
		          "trsm m=%d n=%d ord=%d sd=%d up=%d ta=%d dg=%d",
		          ( int )m, ( int )n, order, side, uplo, trans, diag );

		dim_t lda, ldb;
		double *a_base, *b_base;
		double* a = cmat_alloc( order, dim, dim, &lda, &a_base );
		double* b = cmat_alloc( order, m,   n,   &ldb, &b_base );

		// Diagonally dominant triangular A stays well conditioned; the
		// unreferenced triangle and unit diagonals keep marker bits.
		rng_seed( ( uint32_t )( m * 19 + n ) + ( uint32_t )dim + 29u );
		for ( dim_t i = 0; i < dim; ++i )
		for ( dim_t j = 0; j < dim; ++j )
		{
			const int used = uplo == CblasUpper ? j >= i : i >= j;
			if ( !used )
			{
				// Unused positions read as zero in the oracle.
				mset( a, order, lda, i, j, 0.0 );
			}
			else if ( i == j && diag == CblasUnit )
			{
				set_bits( a + ( order == CblasRowMajor ? i*lda + j : i + j*lda ),
				          CNAN_BITS );
			}
			else if ( i == j ) mset( a, order, lda, i, j, 1.0 + 0.125*( double )( i + 1 ) );
			else               mset( a, order, lda, i, j, 0.125 * rng_val() );
		}

		const size_t na = cmat_size( order, lda, dim, dim );
		double* a_sv = malloc( na * sizeof( double ) );
		double* borg = malloc( (size_t)m * n * sizeof( double ) );
		if ( !a_sv || !borg ) { fprintf( stderr, "out of memory\n" ); exit( 2 ); }
		for ( dim_t i = 0; i < m; ++i )
		for ( dim_t j = 0; j < n; ++j )
		{
			const double v = rng_val();
			mset( b, order, ldb, i, j, v );
			borg[ i*n + j ] = v;
		}
		memcpy( a_sv, a, na * sizeof( double ) );

		const double alpha_v = dg ? 1.2 : -0.75;
		cblas_dtrsm( order, side, uplo, trans, diag, m, n, alpha_v,
		             a, lda, b, ldb );

		CHECK( memcmp( a, a_sv, na * sizeof( double ) ) == 0,
		       "trsm modified A" );

		// Residual check: op(A)*X (Left) or X*op(A) (Right) must equal
		// alpha * B_original elementwise.
		for ( dim_t i = 0; i < m; ++i )
		for ( dim_t j = 0; j < n; ++j )
		{
			double sum = 0.0, absum = 0.0;
			for ( dim_t t = 0; t < dim; ++t )
			{
				// Element (r,c) of the triangular op(A) operand.
				const dim_t r = side == CblasLeft ? i : t;
				const dim_t c = side == CblasLeft ? t : j;
				// Undo the trans flag to find the stored position.
				const dim_t sr = trans == CblasNoTrans ? r : c;
				const dim_t sc = trans == CblasNoTrans ? c : r;
				const int in_tri = uplo == CblasUpper ? sc >= sr : sr >= sc;
				double av;
				if      ( !in_tri )                                av = 0.0;
				else if ( sr == sc && diag == CblasUnit )          av = 1.0;
				else av = mget( a, order, lda, sr, sc );
				const double xv = side == CblasLeft
				  ? mget( b, order, ldb, t, j ) : mget( b, order, ldb, i, t );
				const double term = av * xv;
				sum   += term;
				absum += fabs( term );
			}
			const double expect = alpha_v * borg[ i*n + j ];
			const double bound  = 64.0 * DBL_EPSILON * ( ( double )dim + 2.0 )
			  * ( 1.0 + absum + fabs( expect ) );
			if ( !isfinite( sum ) || fabs( sum - expect ) > bound )
				fail( what );
		}

		cmat_check_edges( "trsm wrote outside logical B", b, order, ldb, m, n );
		check_guards( "trsm overwrote a B guard region", b,
		              cmat_size( order, ldb, m, n ), 0 );

		free( a_sv ); free( borg );
		free( a_base ); free( b_base );
	}
}

// -- complex dispatch paths -----------------------------------------------------------

// Complex matrix helpers on dcomplex storage with a padded leading
// dimension (in complex elements).
static void zset( dcomplex* p, int order, dim_t ld, dim_t r, dim_t c,
                  double re, double im )
{
	dcomplex* e = order == CblasRowMajor ? p + r*ld + c : p + r + c*ld;
	e->real = re;
	e->imag = im;
}

static dcomplex zget( const dcomplex* p, int order, dim_t ld, dim_t r, dim_t c )
{
	return order == CblasRowMajor ? p[ r*ld + c ] : p[ r + c*ld ];
}

static dcomplex* zmat_alloc( int order, dim_t rows, dim_t cols,
                             dim_t* ldp, dcomplex** base_out )
{
	const dim_t  ld   = ( order == CblasRowMajor ? cols : rows ) + 3;
	const size_t n_el = (size_t)ld * ( order == CblasRowMajor ? rows : cols );
	*ldp = ld;
	return ( dcomplex* )galloc( 2 * n_el, 0, ( double** )base_out );
}

static void test_cblas_zgemm( void )
{
	static const dim_t SHAPES[][3] =
		{ { 3, 5, 7 }, { 4, 4, 4 }, { 5, 3, 9 }, { 8, 9, 6 } };
	static const int ORDERS[] = { CblasRowMajor, CblasColMajor };
	static const int TRANS[]  = { CblasNoTrans, CblasTrans, CblasConjTrans };

	// Run every case twice: once on the native dcomplex path and once
	// with the induced real-precision (1m) method forced on. The latter
	// lowers z gemm onto the real double microkernel this test covers.
	for ( int im = 0; im < 2; ++im )
	{
		if ( im ) bli_ind_enable_dt( BLIS_1M, BLIS_DCOMPLEX );
		else      bli_ind_disable_dt( BLIS_1M, BLIS_DCOMPLEX );

		for ( size_t is = 0; is < sizeof( SHAPES )/sizeof( SHAPES[0] ); ++is )
		for ( int io = 0; io < 2; ++io )
		for ( int ta = 0; ta < 3; ++ta )
		for ( int tb = 0; tb < 3; ++tb )
		{
			const dim_t m = SHAPES[is][0], n = SHAPES[is][1], k = SHAPES[is][2];
			const int order = ORDERS[io], transa = TRANS[ta], transb = TRANS[tb];

			char what[160];
			snprintf( what, sizeof( what ),
			          "zgemm m=%d n=%d k=%d ord=%d ta=%d tb=%d m1=%d",
			          ( int )m, ( int )n, ( int )k, order, transa, transb, im );

			const dim_t ar = transa == CblasNoTrans ? m : k;
			const dim_t ac = transa == CblasNoTrans ? k : m;
			const dim_t br = transb == CblasNoTrans ? k : n;
			const dim_t bc = transb == CblasNoTrans ? n : k;

			dim_t lda, ldb, ldc;
			dcomplex *a_base, *b_base, *c_base;
			dcomplex* a = zmat_alloc( order, ar, ac, &lda, &a_base );
			dcomplex* b = zmat_alloc( order, br, bc, &ldb, &b_base );
			dcomplex* c = zmat_alloc( order, m,  n,  &ldc, &c_base );

			rng_seed( ( uint32_t )( m * 41 + n * 13 + k ) + 43u );
			for ( dim_t i = 0; i < ar; ++i )
			for ( dim_t j = 0; j < ac; ++j )
				zset( a, order, lda, i, j, rng_val(), rng_val() );
			for ( dim_t i = 0; i < br; ++i )
			for ( dim_t j = 0; j < bc; ++j )
				zset( b, order, ldb, i, j, rng_val(), rng_val() );

			dcomplex* clog = malloc( (size_t)m * n * sizeof( dcomplex ) );
			if ( !clog ) { fprintf( stderr, "out of memory\n" ); exit( 2 ); }
			for ( dim_t i = 0; i < m; ++i )
			for ( dim_t j = 0; j < n; ++j )
			{
				const dcomplex v = { rng_val(), rng_val() };
				zset( c, order, ldc, i, j, v.real, v.imag );
				clog[ i*n + j ] = v;
			}

			const dcomplex alpha_v = { 1.2, -0.5 };
			const dcomplex beta_v  = { -0.75, 0.25 };

			cblas_zgemm( order, transa, transb, m, n, k,
			             &alpha_v, a, lda, b, ldb, &beta_v, c, ldc );

			for ( dim_t i = 0; i < m; ++i )
			for ( dim_t j = 0; j < n; ++j )
			{
				dcomplex sum = { 0.0, 0.0 };
				double   absum = 0.0;
				for ( dim_t p = 0; p < k; ++p )
				{
					dcomplex av = transa == CblasNoTrans
					  ? zget( a, order, lda, i, p ) : zget( a, order, lda, p, i );
					dcomplex bv = transb == CblasNoTrans
					  ? zget( b, order, ldb, p, j ) : zget( b, order, ldb, j, p );
					if ( transa == CblasConjTrans ) av.imag = -av.imag;
					if ( transb == CblasConjTrans ) bv.imag = -bv.imag;
					const dcomplex t =
						{ av.real*bv.real - av.imag*bv.imag,
						  av.real*bv.imag + av.imag*bv.real };
					sum.real += t.real;
					sum.imag += t.imag;
					absum    += fabs( t.real ) + fabs( t.imag );
				}
				const dcomplex as =
					{ alpha_v.real*sum.real - alpha_v.imag*sum.imag,
					  alpha_v.real*sum.imag + alpha_v.imag*sum.real };
				const dcomplex bc2 = clog[ i*n + j ];
				const dcomplex bs =
					{ beta_v.real*bc2.real - beta_v.imag*bc2.imag,
					  beta_v.real*bc2.imag + beta_v.imag*bc2.real };
				const double er = as.real + bs.real;
				const double ei = as.imag + bs.imag;
				const double bound = 48.0 * DBL_EPSILON * ( ( double )k + 2.0 )
				  * ( 1.0 + absum * 1.6 + fabs( er ) + fabs( ei ) );
				const dcomplex got = zget( c, order, ldc, i, j );
				if ( !isfinite( got.real ) || !isfinite( got.imag ) ||
				     fabs( got.real - er ) > bound ||
				     fabs( got.imag - ei ) > bound )
					fail( what );
			}

			free( clog );
			free( a_base ); free( b_base ); free( c_base );
		}
	}

	// Restore the method flag to its initialized state.
	bli_ind_enable_dt( BLIS_1M, BLIS_DCOMPLEX );
}

// -- dispatch assertions -----------------------------------------------------------

// Compiled only when the wasm32 kernel set is registered; the same source
// must still pass on a pure reference build.
#ifdef BLIS_KERNELS_WASM32

static gemm_ukr_ft saved_dgemm_ukr;
static unsigned    dgemm_ukr_calls;

// Counting delegate proving that public gemm calls reach the selected
// microkernel; installed only for the duration of the probe.
static void counting_dgemm_ukr
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	++dgemm_ukr_calls;
	saved_dgemm_ukr( m, n, k, alpha, a, b, beta, c, rs_c, cs_c, data, cntx );
}

static void test_dispatch( const cntx_t* cntx )
{
	const func_t* gemm_ukrs = bli_cntx_get_ukrs( BLIS_GEMM_UKR, cntx );
	const void_fp dgemm_ukr = bli_func_get_dt( BLIS_DOUBLE, gemm_ukrs );

	CHECK( dgemm_ukr == ( void_fp )bli_dgemm_wasm32_simd128_4x4,
	       "double gemm ukr is not the wasm32 simd128 kernel" );
	CHECK( bli_cntx_get_ukr_prefs_dt( BLIS_DOUBLE, BLIS_GEMM_UKR_ROW_PREF, cntx ),
	       "double gemm ukr row preference not set" );

	CHECK( bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MR, cntx ) == 4 &&
	       bli_cntx_get_blksz_max_dt( BLIS_DOUBLE, BLIS_MR, cntx ) == 4,
	       "double MR is not 4/4" );
	CHECK( bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NR, cntx ) == 4 &&
	       bli_cntx_get_blksz_max_dt( BLIS_DOUBLE, BLIS_NR, cntx ) == 4,
	       "double NR is not 4/4" );
	CHECK( bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_KR,  cntx ) == 1,
	       "double KR is not 1" );
	CHECK( bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_BBM, cntx ) == 1 &&
	       bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_BBN, cntx ) == 1,
	       "double BBM/BBN are not 1" );
	CHECK( bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MC, cntx ) == 128 &&
	       bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_KC, cntx ) == 256 &&
	       bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NC, cntx ) == 4096,
	       "double cache blocksizes changed" );
	CHECK( bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MT, cntx ) == 0 &&
	       bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NT, cntx ) == 0 &&
	       bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_KT, cntx ) == 0,
	       "double sup thresholds changed" );

	// Every other datatype and kernel slot must match a fresh reference
	// context for this architecture.
	cntx_t ref;
	bli_gks_init_ref_cntx( &ref );

	static const kerid_t ukrs[] =
	{
		BLIS_GEMM_UKR, BLIS_GEMM1M_UKR,
		BLIS_GEMMTRSM_L_UKR, BLIS_GEMMTRSM_U_UKR,
		BLIS_TRSM_L_UKR,     BLIS_TRSM_U_UKR,
		BLIS_GEMMTRSM1M_L_UKR, BLIS_GEMMTRSM1M_U_UKR,
	};
	static const num_t dts[] =
		{ BLIS_FLOAT, BLIS_DOUBLE, BLIS_SCOMPLEX, BLIS_DCOMPLEX };

	for ( size_t i = 0; i < sizeof( ukrs )/sizeof( ukrs[0] ); ++i )
	for ( size_t j = 0; j < sizeof( dts ) / sizeof( dts[0] ); ++j )
	{
		if ( ukrs[i] == BLIS_GEMM_UKR && dts[j] == BLIS_DOUBLE ) continue;
		const func_t* f_new = bli_cntx_get_ukrs( ukrs[i], cntx );
		const func_t* f_ref = bli_cntx_get_ukrs( ukrs[i], &ref );
		CHECK( bli_func_get_dt( dts[j], f_new ) ==
		       bli_func_get_dt( dts[j], f_ref ),
		       "an unmodified ukr slot changed" );
	}

	static const bszid_t bss[] =
	{
		BLIS_KR, BLIS_MR, BLIS_NR, BLIS_MC, BLIS_KC, BLIS_NC,
		BLIS_BBM, BLIS_BBN, BLIS_M2, BLIS_N2, BLIS_AF, BLIS_DF, BLIS_XF,
		BLIS_MT, BLIS_NT, BLIS_KT,
		BLIS_KR_SUP, BLIS_MR_SUP, BLIS_NR_SUP,
		BLIS_MC_SUP, BLIS_KC_SUP, BLIS_NC_SUP,
	};
	for ( size_t i = 0; i < sizeof( bss )/sizeof( bss[0] ); ++i )
	for ( size_t j = 0; j < sizeof( dts ) / sizeof( dts[0] ); ++j )
	{
		if ( dts[j] == BLIS_DOUBLE &&
		     ( bss[i] == BLIS_MR || bss[i] == BLIS_NR ) ) continue;
		CHECK( bli_cntx_get_blksz_def_dt( dts[j], bss[i], cntx ) ==
		       bli_cntx_get_blksz_def_dt( dts[j], bss[i], &ref ) &&
		       bli_cntx_get_blksz_max_dt( dts[j], bss[i], cntx ) ==
		       bli_cntx_get_blksz_max_dt( dts[j], bss[i], &ref ),
		       "an unmodified blocksize changed" );
	}

	bli_cntx_free( &ref );

	// Public dispatch probe: wrap the live double ukr, run a nontrivial
	// public gemm, then restore the original pointer before returning.
	cntx_t* mcntx = ( cntx_t* )cntx;
	saved_dgemm_ukr = ( gemm_ukr_ft )dgemm_ukr;
	dgemm_ukr_calls = 0;
	bli_cntx_set_ukr_dt( ( void_fp )counting_dgemm_ukr, BLIS_DOUBLE,
	                     BLIS_GEMM_UKR, mcntx );

	{
		double a[17*23], b[23*19], c[17*19];
		rng_seed( 7u );
		for ( size_t i = 0; i < sizeof( a )/sizeof( a[0] ); ++i ) a[i] = rng_val();
		for ( size_t i = 0; i < sizeof( b )/sizeof( b[0] ); ++i ) b[i] = rng_val();
		for ( size_t i = 0; i < sizeof( c )/sizeof( c[0] ); ++i ) c[i] = rng_val();

		cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
		             17, 19, 23, 1.0, a, 17, b, 23, 0.0, c, 17 );

		CHECK( dgemm_ukr_calls > 0,
		       "public cblas_dgemm did not reach the double ukr" );
	}

	bli_cntx_set_ukr_dt( ( void_fp )saved_dgemm_ukr, BLIS_DOUBLE,
	                     BLIS_GEMM_UKR, mcntx );
}

#endif // BLIS_KERNELS_WASM32

int main( void )
{
	bli_init();

	const cntx_t* cntx = bli_gks_query_cntx();
	CHECK( cntx != NULL, "no native context" );

	test_ukr( cntx );
	test_cblas_gemm();
	test_cblas_trsm();
	test_cblas_zgemm();

#ifdef BLIS_KERNELS_WASM32
	test_dispatch( cntx );
#endif

	printf( "test-dgemm: %lu checks, %lu failures\n", n_check, n_fail );

	bli_finalize();
	return n_fail ? 1 : 0;
}
