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

// Contract test for the trsm microkernels registered for one real
// datatype, and for the public cblas_?trsm calls that reach them through
// the reference gemmtrsm microkernel. The including file defines
// test_type, test_bits (an unsigned integer of the same size), TEST_DT,
// TEST_EPS, TEST_MARK (a signaling NaN bit pattern), TEST_NAME and
// TEST_TRSM. It also defines TEST_KERNEL_L, TEST_KERNEL_U, TEST_MR and
// TEST_NR when an optimized pair with that tile must be registered;
// otherwise the reference pair must be. Register and packing sizes are
// taken from the runtime context. Exits nonzero when any check fails.

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "blis.h"
#include "cblas.h"

// -- accounting ---------------------------------------------------------------

static unsigned long n_check, n_fail;

static void check( int ok, const char* problem, const char* what )
{
	++n_check;
	if ( !ok )
	{
		if ( n_fail < 25 ) fprintf( stderr, "FAIL: %s: %s\n", problem, what );
		++n_fail;
	}
}

static void* xmalloc( size_t bytes )
{
	void* p = malloc( bytes ? bytes : 1 );
	if ( !p ) { fprintf( stderr, "out of memory\n" ); exit( 2 ); }
	return p;
}

// -- deterministic values -------------------------------------------------------

static uint32_t rng_state;

static void rng_seed( uint32_t s ) { rng_state = s ? s : 1u; }

static int rng_int( int n )
{
	rng_state = rng_state * 1664525u + 1013904223u;
	return ( int )( ( rng_state >> 8 ) % ( uint32_t )n );
}

// Multiples of 0.25 in [-2, 2].
static double rng_quarter( void ) { return ( rng_int( 17 ) - 8 ) * 0.25; }

// 0, +-0.5 or +-1.
static double rng_half( void ) { return ( rng_int( 5 ) - 2 ) * 0.5; }

// +-0.5, +-1 or +-2.
static double rng_pow2( void )
{
	static const double v[] = { 0.5, 1.0, 2.0, -0.5, -1.0, -2.0 };
	return v[ rng_int( 6 ) ];
}

// -- markers ----------------------------------------------------------------------

// Bytes that must stay untouched, and elements that must not be used, hold
// a signaling NaN. Arithmetic on it gives a quiet NaN with different bits,
// so a value computed from a marker never passes for the marker or for data.
static void set_mark( test_type* p )
{
	const test_bits u = TEST_MARK;
	memcpy( p, &u, sizeof( u ) );
}

static int is_mark( const test_type* p )
{
	test_bits u;
	memcpy( &u, p, sizeof( u ) );
	return u == ( test_bits )TEST_MARK;
}

#define GUARD 8

// n elements between GUARD marker elements on each side, all marked.
static test_type* galloc( size_t n, test_type** base )
{
	*base = xmalloc( ( n + 2*GUARD ) * sizeof( test_type ) );
	for ( size_t s = 0; s < n + 2*GUARD; ++s ) set_mark( *base + s );
	return *base + GUARD;
}

static int guards_intact( const test_type* base, size_t n )
{
	for ( size_t s = 0; s < GUARD; ++s )
		if ( !is_mark( base + s ) || !is_mark( base + GUARD + n + s ) ) return 0;
	return 1;
}

// The packed diagonal holds the inverse of each diagonal element when trsm
// preinversion is enabled, and the element itself otherwise.
static long double apply_diag( long double s, test_type d )
{
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	return s * d;
#else
	return s / d;
#endif
}

// -- direct microkernel calls ---------------------------------------------------

typedef struct
{
	const cntx_t* cntx;
	dim_t         mr, nr;   // register tile
	dim_t         pmr, pnr; // leading dimensions of packed A11 and B11
} env_t;

enum { K_EXACT, K_UNIT, K_EDGE, K_GENERAL, K_COUNT };

static const char* const kind_name[ K_COUNT ] =
	{ "exact", "unit-diagonal", "edge-padded", "general" };

// Call one trsm microkernel on hand-packed A11 and B11 and check the
// solution in both B11 and C11. Exact and edge data are dyadic, so their
// solution must match the extended-precision oracle exactly.
static void ukr_case( const env_t* e, void_fp ukr_fp, int upper, int kind,
                      inc_t rs, inc_t cs, int skew, uint32_t seed )
{
	const trsm_ukr_ft ukr = ( trsm_ukr_ft )ukr_fp;
	const dim_t mr = e->mr, nr = e->nr, pmr = e->pmr, pnr = e->pnr;

	char what[ 160 ];
	snprintf( what, sizeof( what ), "trsm_%c ukr, %s data, rs=%d cs=%d skew=%d seed=%u",
	          upper ? 'u' : 'l', kind_name[ kind ], ( int )rs, ( int )cs, skew,
	          ( unsigned )seed );

	// A11 and B11 get exact-size allocations, so AddressSanitizer reports
	// any access past either packed block.
	const size_t na = ( size_t )pmr * mr;
	const size_t nb = ( size_t )mr * pnr;
	test_type*   a_base = xmalloc( ( na + skew ) * sizeof( test_type ) );
	test_type*   b_base = xmalloc( ( nb + skew ) * sizeof( test_type ) );
	test_type*   a_copy = xmalloc( na * sizeof( test_type ) );
	long double* b0     = xmalloc( ( size_t )mr * nr * sizeof( long double ) );
	long double* x      = xmalloc( ( size_t )mr * nr * sizeof( long double ) );
	test_type*   a      = a_base + skew;
	test_type*   b      = b_base + skew;

	// Edge data mimics what packing produces for a diagonal block with
	// fewer than mr rows: rows and columns past m_used are zero in A11
	// except for a unit diagonal, and those rows of B11 are zero.
	const dim_t m_used = kind == K_EDGE && mr > 1
	                     ? 1 + ( dim_t )( seed % ( uint32_t )( mr - 1 ) ) : mr;

	// A11 is stored by columns with leading dimension pmr. The unstored
	// triangle and any rows past mr hold the marker: a kernel that uses
	// them produces NaNs.
	rng_seed( seed );
	for ( dim_t l = 0; l < mr; ++l )
	for ( dim_t i = 0; i < pmr; ++i )
	{
		test_type* p = a + i + l*pmr;
		if ( i >= mr || ( upper ? l < i : l > i ) ) { set_mark( p ); continue; }
		if ( i == l )
		{
			double d;
			if ( kind == K_UNIT || i >= m_used ) d = 1.0;
			else if ( kind == K_GENERAL )
			{
				const double mag = 1.0 + rng_int( 7 ) / 3.0;
				d = rng_int( 2 ) ? mag : -mag;
			}
			else d = rng_pow2();
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
			*p = ( test_type )( 1.0 / d );
#else
			*p = ( test_type )d;
#endif
		}
		else if ( i >= m_used || l >= m_used ) *p = 0;
		else if ( kind == K_GENERAL )          *p = ( test_type )( rng_quarter() / 3.0 );
		else                                   *p = ( test_type )rng_half();
	}

	// B11 is stored by rows with leading dimension pnr.
	for ( dim_t i = 0; i < mr; ++i )
	for ( dim_t j = 0; j < pnr; ++j )
	{
		test_type* p = b + i*pnr + j;
		if ( j >= nr ) { set_mark( p ); continue; }
		if      ( i >= m_used )       *p = 0;
		else if ( kind == K_GENERAL ) *p = ( test_type )( rng_quarter() / 3.0 + 0.125 );
		else                          *p = ( test_type )rng_quarter();
		b0[ i*nr + j ] = *p;
	}
	memcpy( a_copy, a, na * sizeof( test_type ) );

	// C11 span: with negative strides the tile does not start at c.
	inc_t lo = 0, hi = 0;
	for ( dim_t i = 0; i < mr; ++i )
	for ( dim_t j = 0; j < nr; ++j )
	{
		const inc_t off = i*rs + j*cs;
		if ( off < lo ) lo = off;
		if ( off > hi ) hi = off;
	}
	const size_t   span = ( size_t )( hi - lo + 1 );
	test_type*     c_base;
	test_type*     c    = galloc( span + skew, &c_base ) + skew - lo;
	unsigned char* tile = calloc( span, 1 );
	if ( !tile ) { fprintf( stderr, "out of memory\n" ); exit( 2 ); }
	for ( dim_t i = 0; i < mr; ++i )
	for ( dim_t j = 0; j < nr; ++j )
		tile[ i*rs + j*cs - lo ] = 1;

	auxinfo_t aux;
	memset( &aux, 0, sizeof( aux ) );
	bli_auxinfo_set_next_a( a, &aux );
	bli_auxinfo_set_next_b( b, &aux );

	ukr( a, b, c, rs, cs, &aux, e->cntx );

	check( memcmp( a, a_copy, na * sizeof( test_type ) ) == 0,
	       "packed A11 was modified", what );

	// Substitution oracle on the original B11 in extended precision, with
	// the packed diagonal applied as the contract defines it.
	for ( dim_t j = 0; j < nr; ++j )
	for ( dim_t it = 0; it < mr; ++it )
	{
		const dim_t i  = upper ? mr - 1 - it : it;
		const dim_t l0 = upper ? i + 1 : 0;
		const dim_t l1 = upper ? mr : i;
		long double s = b0[ i*nr + j ];
		for ( dim_t l = l0; l < l1; ++l )
			s -= ( long double )a[ i + l*pmr ] * x[ l*nr + j ];
		x[ i*nr + j ] = apply_diag( s, a[ i + i*pmr ] );
	}

	for ( dim_t i = 0; i < mr; ++i )
	for ( dim_t j = 0; j < nr; ++j )
	{
		const test_type got = b[ i*pnr + j ];
		check( memcmp( &got, c + i*rs + j*cs, sizeof( got ) ) == 0,
		       "C11 differs from the updated B11", what );
		if ( kind != K_GENERAL )
		{
			check( isfinite( got ) && ( long double )got == x[ i*nr + j ],
			       "result differs from the exact solution", what );
			continue;
		}

		// Recompute row i from the kernel's own rows l, so that the bound
		// only has to cover the rounding of one row update.
		const dim_t l0 = upper ? i + 1 : 0;
		const dim_t l1 = upper ? mr : i;
		long double s = b0[ i*nr + j ], mag = fabsl( s );
		for ( dim_t l = l0; l < l1; ++l )
		{
			const long double t = ( long double )a[ i + l*pmr ] * b[ l*pnr + j ];
			s   -= t;
			mag += fabsl( t );
		}
		const long double want  = apply_diag( s, a[ i + i*pmr ] );
		const long double scale = fabsl( apply_diag( 1.0L, a[ i + i*pmr ] ) );
		const long double bound = 4.0L * ( mr + 2 ) * TEST_EPS * mag * scale;
		check( isfinite( got ) && fabsl( got - want ) <= bound,
		       "result outside the rounding bound", what );
	}

	// Only the tile may change: every other slot of the C span, the guards
	// and the B11 columns past nr keep the marker.
	int intact = guards_intact( c_base, span + skew );
	for ( int s = 0; s < skew; ++s )
		if ( !is_mark( c_base + GUARD + s ) ) intact = 0;
	for ( size_t s = 0; s < span; ++s )
		if ( !tile[ s ] && !is_mark( c + lo + ( inc_t )s ) ) intact = 0;
	for ( dim_t i = 0; i < mr; ++i )
	for ( dim_t j = nr; j < pnr; ++j )
		if ( !is_mark( b + i*pnr + j ) ) intact = 0;
	check( intact, "wrote outside the C11 tile or the B11 block", what );

	free( tile ); free( c_base );
	free( a_base ); free( b_base ); free( a_copy );
	free( b0 ); free( x );
}

static void test_ukr( const env_t* e, void_fp ukr_l, void_fp ukr_u )
{
	const inc_t mr = e->mr, nr = e->nr;
	const inc_t lays[][2] =
	{
		{ nr,             1 },    // the caller's temporary tile for edges
		{ nr + 3,         1 },    // row-stored with gaps
		{ 1,             mr },    // column-stored, contiguous
		{ 1,         mr + 3 },    // column-stored with gaps
		{ 2*nr + 3,       2 },    // general strides
		{ 2,       2*mr + 1 },    // general strides, columns farther apart
		{ -( nr + 3 ),    1 },    // reversed rows
		{ 1,    -( mr + 3 ) },    // reversed columns
		{ -( 2*nr + 3 ), -2 },    // both reversed
	};
	const int nlays = ( int )( sizeof( lays ) / sizeof( lays[0] ) );

	uint32_t seed = 1;
	for ( int upper = 0; upper < 2; ++upper )
	for ( int kind = 0; kind < K_COUNT; ++kind )
	for ( int il = 0; il < nlays; ++il )
	for ( int skew = 0; skew < 2; ++skew )
	for ( int rep = 0; rep < 3; ++rep )
		ukr_case( e, upper ? ukr_u : ukr_l, upper, kind,
		          lays[il][0], lays[il][1], skew, seed++ );
}

// -- public cblas_?trsm -------------------------------------------------------------

// Counting delegates for the live trsm microkernels. The reference gemmtrsm
// microkernel looks the trsm microkernel up in the context on every call,
// so these see every call that a public trsm makes.
static trsm_ukr_ft    live_l, live_u;
static unsigned long  calls_l, calls_u;

static void counting_l( const void* a, void* b, void* c, inc_t rs_c, inc_t cs_c,
                        const auxinfo_t* data, const cntx_t* cntx )
{
	++calls_l;
	live_l( a, b, c, rs_c, cs_c, data, cntx );
}

static void counting_u( const void* a, void* b, void* c, inc_t rs_c, inc_t cs_c,
                        const auxinfo_t* data, const cntx_t* cntx )
{
	++calls_u;
	live_u( a, b, c, rs_c, cs_c, data, cntx );
}

static size_t at( int order, dim_t ld, dim_t r, dim_t c )
{
	return order == CblasRowMajor ? ( size_t )( r*ld + c ) : ( size_t )( r + c*ld );
}

static void public_case( int order, int side, int uplo, int trans, int diag,
                         dim_t m, dim_t n, double alpha, uint32_t seed )
{
	char what[ 160 ];
	snprintf( what, sizeof( what ),
	          "cblas trsm m=%d n=%d %s side=%c uplo=%c trans=%c diag=%c alpha=%g",
	          ( int )m, ( int )n, order == CblasRowMajor ? "row-major" : "col-major",
	          side == CblasLeft ? 'L' : 'R', uplo == CblasLower ? 'L' : 'U',
	          trans == CblasNoTrans ? 'N' : trans == CblasTrans ? 'T' : 'C',
	          diag == CblasUnit ? 'U' : 'N', alpha );

	const dim_t  dim  = side == CblasLeft ? m : n;
	const dim_t  lda  = dim + 3;
	const dim_t  ldb  = ( order == CblasRowMajor ? n : m ) + 3;
	const dim_t  nvec = order == CblasRowMajor ? m : n;
	const size_t na   = ( size_t )lda * dim;
	const size_t nb   = ( size_t )ldb * nvec;

	test_type *a_base, *b_base;
	test_type* a      = galloc( na, &a_base );
	test_type* b      = galloc( nb, &b_base );
	test_type* a_copy = xmalloc( ( na + 2*GUARD ) * sizeof( test_type ) );
	double*    aval   = xmalloc( ( size_t )dim * dim * sizeof( double ) );
	double*    b0     = xmalloc( ( size_t )m * n * sizeof( double ) );

	// Off-diagonal entries are small multiples of a power of two and the
	// diagonal is +-1 or +-2, so every row and column of A is diagonally
	// dominant and the solution stays well scaled at every size. The
	// unstored triangle, a unit diagonal and the leading-dimension padding
	// keep the marker: trsm must not read them.
	double scale = 0.5;
	while ( scale * ( dim + 1 ) > 0.25 ) scale *= 0.5;
	rng_seed( seed );
	for ( dim_t r = 0; r < dim; ++r )
	for ( dim_t c = 0; c < dim; ++c )
	{
		const int stored = uplo == CblasLower ? c <= r : c >= r;
		double v = 0.0;
		if ( stored && r == c )
		{
			if ( diag == CblasUnit ) v = 1.0;
			else
			{
				v = rng_int( 2 ) ? 2.0 : 1.0;
				if ( rng_int( 2 ) ) v = -v;
				a[ at( order, lda, r, c ) ] = ( test_type )v;
			}
		}
		else if ( stored )
		{
			v = rng_quarter() * scale;
			a[ at( order, lda, r, c ) ] = ( test_type )v;
		}
		aval[ r*dim + c ] = v;
	}
	memcpy( a_copy, a_base, ( na + 2*GUARD ) * sizeof( test_type ) );

	for ( dim_t i = 0; i < m; ++i )
	for ( dim_t j = 0; j < n; ++j )
	{
		const double v = rng_quarter();
		b[ at( order, ldb, i, j ) ] = ( test_type )v;
		b0[ i*n + j ] = v;
	}

	calls_l = calls_u = 0;
	TEST_TRSM( order, side, uplo, trans, diag, m, n, ( test_type )alpha,
	           a, lda, b, ldb );

	check( memcmp( a_base, a_copy, ( na + 2*GUARD ) * sizeof( test_type ) ) == 0,
	       "A was modified", what );

	// A right-side solve runs as a left-side solve with op(A) transposed,
	// so trsm_l handles a lower op(A) on the left and an upper op(A) on
	// the right.
	const int lower_op = ( uplo == CblasLower ) == ( trans == CblasNoTrans );
	const int use_l    = side == CblasLeft ? lower_op : !lower_op;
	if ( m > 0 && n > 0 && alpha != 0.0 )
	{
		check( ( use_l ? calls_l : calls_u ) > 0,
		       "the expected trsm microkernel was not called", what );
		check( ( use_l ? calls_u : calls_l ) == 0,
		       "the other trsm microkernel was called", what );
	}
	else check( calls_l == 0 && calls_u == 0,
	            "a trivial call reached a trsm microkernel", what );

	// Residual of op(A) X = alpha B (left) or X op(A) = alpha B (right).
	// trsm is backward stable, so the residual is bounded by a small
	// multiple of eps times |op(A)| |X|.
	for ( dim_t i = 0; i < m; ++i )
	for ( dim_t j = 0; j < n; ++j )
	{
		const double xij    = b[ at( order, ldb, i, j ) ];
		const double expect = alpha * b0[ i*n + j ];
		if ( alpha == 0.0 )
		{
			check( xij == 0.0, "alpha = 0 did not zero B", what );
			continue;
		}
		double sum = 0.0, mag = fabs( expect );
		for ( dim_t t = 0; t < dim; ++t )
		{
			const dim_t  r  = side == CblasLeft ? i : t;
			const dim_t  c  = side == CblasLeft ? t : j;
			const double av = trans == CblasNoTrans ? aval[ r*dim + c ]
			                                        : aval[ c*dim + r ];
			if ( av == 0.0 ) continue;
			const double xv = side == CblasLeft ? b[ at( order, ldb, t, j ) ]
			                                    : b[ at( order, ldb, i, t ) ];
			sum += av * xv;
			mag += fabs( av * xv );
		}
		check( isfinite( xij ) &&
		       fabs( sum - expect ) <= 32.0 * TEST_EPS * ( dim + 2 ) * mag,
		       "residual of the solution too large", what );
	}

	// Leading-dimension padding and guards keep the marker.
	int intact = guards_intact( b_base, nb );
	for ( dim_t v = 0; v < nvec; ++v )
	for ( dim_t s = ( order == CblasRowMajor ? n : m ); s < ldb; ++s )
		if ( !is_mark( b + v*ldb + s ) ) intact = 0;
	check( intact, "wrote outside the logical B", what );

	free( a_base ); free( b_base ); free( a_copy );
	free( aval ); free( b0 );
}

static void test_public( cntx_t* cntx )
{
	// Sizes around mr and nr, plus triangular dimensions above MC (129
	// for double, 259 for float) and above KC (259) for both sides.
	static const dim_t shapes[][2] =
	{
		{ 1, 1 }, { 1, 5 }, { 5, 1 }, { 3, 5 }, { 4, 4 }, { 5, 3 },
		{ 7, 9 }, { 9, 7 }, { 8, 8 }, { 17, 19 }, { 33, 6 }, { 6, 33 },
		{ 129, 6 }, { 6, 129 }, { 259, 5 }, { 5, 259 },
	};
	static const int orders[] = { CblasRowMajor, CblasColMajor };
	static const int sides[]  = { CblasLeft, CblasRight };
	static const int uplos[]  = { CblasLower, CblasUpper };
	static const int transs[] = { CblasNoTrans, CblasTrans, CblasConjTrans };
	static const int diags[]  = { CblasNonUnit, CblasUnit };
	static const double alphas[] = { 1.0, -0.75, 2.0 };

	live_l = ( trsm_ukr_ft )bli_cntx_get_ukr_dt( TEST_DT, BLIS_TRSM_L_UKR, cntx );
	live_u = ( trsm_ukr_ft )bli_cntx_get_ukr_dt( TEST_DT, BLIS_TRSM_U_UKR, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )counting_l, TEST_DT, BLIS_TRSM_L_UKR, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )counting_u, TEST_DT, BLIS_TRSM_U_UKR, cntx );

	uint32_t combo = 0;
	for ( size_t is = 0; is < sizeof( shapes ) / sizeof( shapes[0] ); ++is )
	for ( int io = 0; io < 2; ++io )
	for ( int sd = 0; sd < 2; ++sd )
	for ( int up = 0; up < 2; ++up )
	for ( int tr = 0; tr < 3; ++tr )
	for ( int dg = 0; dg < 2; ++dg )
	{
		public_case( orders[io], sides[sd], uplos[up], transs[tr], diags[dg],
		             shapes[is][0], shapes[is][1], alphas[ combo % 3 ], 101u + combo );
		++combo;
	}

	// Empty B and alpha = 0 return before any microkernel runs.
	for ( int io = 0; io < 2; ++io )
	for ( int sd = 0; sd < 2; ++sd )
	{
		public_case( orders[io], sides[sd], CblasLower, CblasNoTrans, CblasNonUnit,
		             0, 5, 1.0, 7u );
		public_case( orders[io], sides[sd], CblasUpper, CblasTrans, CblasUnit,
		             5, 0, 1.0, 8u );
		public_case( orders[io], sides[sd], CblasLower, CblasNoTrans, CblasNonUnit,
		             6, 7, 0.0, 9u );
	}

	bli_cntx_set_ukr_dt( ( void_fp )live_l, TEST_DT, BLIS_TRSM_L_UKR, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )live_u, TEST_DT, BLIS_TRSM_U_UKR, cntx );
}

int main( void )
{
	bli_init();

	cntx_t* cntx = ( cntx_t* )bli_gks_query_cntx();

	env_t e;
	e.cntx = cntx;
	e.mr   = bli_cntx_get_blksz_def_dt( TEST_DT, BLIS_MR, cntx );
	e.nr   = bli_cntx_get_blksz_def_dt( TEST_DT, BLIS_NR, cntx );
	e.pmr  = bli_cntx_get_blksz_max_dt( TEST_DT, BLIS_MR, cntx );
	e.pnr  = bli_cntx_get_blksz_max_dt( TEST_DT, BLIS_NR, cntx );
	check( e.mr > 0 && e.nr > 0 && e.pmr >= e.mr && e.pnr >= e.nr,
	       "invalid register or packing sizes", TEST_NAME );
	check( bli_cntx_get_blksz_def_dt( TEST_DT, BLIS_BBM, cntx ) == 1 &&
	       bli_cntx_get_blksz_def_dt( TEST_DT, BLIS_BBN, cntx ) == 1,
	       "packed elements are broadcast, which the direct cases do not model",
	       TEST_NAME );

	const void_fp ukr_l = bli_cntx_get_ukr_dt( TEST_DT, BLIS_TRSM_L_UKR, cntx );
	const void_fp ukr_u = bli_cntx_get_ukr_dt( TEST_DT, BLIS_TRSM_U_UKR, cntx );

	cntx_t ref;
	bli_gks_init_ref_cntx( &ref );
#ifdef TEST_KERNEL_L
	check( ukr_l == ( void_fp )TEST_KERNEL_L && ukr_u == ( void_fp )TEST_KERNEL_U,
	       "the optimized trsm microkernels are not registered", TEST_NAME );
	check( e.mr == TEST_MR && e.pmr == TEST_MR && e.nr == TEST_NR && e.pnr == TEST_NR,
	       "register or packing sizes differ from the trsm microkernel tile",
	       TEST_NAME );
#else
	check( ukr_l == bli_cntx_get_ukr_dt( TEST_DT, BLIS_TRSM_L_UKR, &ref ) &&
	       ukr_u == bli_cntx_get_ukr_dt( TEST_DT, BLIS_TRSM_U_UKR, &ref ),
	       "the trsm microkernels are not the reference ones", TEST_NAME );
#endif
	// The trsm microkernels are reached through the reference gemmtrsm
	// microkernel, which pairs them with the registered gemm microkernel.
	check( bli_cntx_get_ukr_dt( TEST_DT, BLIS_GEMMTRSM_L_UKR, cntx ) ==
	       bli_cntx_get_ukr_dt( TEST_DT, BLIS_GEMMTRSM_L_UKR, &ref ) &&
	       bli_cntx_get_ukr_dt( TEST_DT, BLIS_GEMMTRSM_U_UKR, cntx ) ==
	       bli_cntx_get_ukr_dt( TEST_DT, BLIS_GEMMTRSM_U_UKR, &ref ),
	       "the gemmtrsm microkernels are not the reference ones", TEST_NAME );
	bli_cntx_free( &ref );

	test_ukr( &e, ukr_l, ukr_u );
	test_public( cntx );

	printf( "%s: %lu checks, %lu failures\n", TEST_NAME, n_check, n_fail );

	bli_finalize();
	return n_fail ? 1 : 0;
}
