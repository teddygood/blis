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

// Contract test for the wasm32 real dotv and axpyf kernels. Every case runs
// the SIMD kernel and the reference kernel of this configuration on the same
// operands and compares the results:
//  - exactly, for inputs that are small multiples of 1/4, whose products and
//    sums are exact in both precisions at these lengths;
//  - within a rounding-error bound, for pseudo-random inputs;
//  - by class (NaN, +Inf or -Inf) when an input holds a NaN or an infinity.
// Inputs are allocated with their exact extent, so an AddressSanitizer build
// catches reads past the end, and must not change. Outputs are surrounded by
// guard elements, and the elements that a non-unit increment skips, which
// must not change either. A last section wraps the registered kernels and
// checks that cblas_?dot and cblas_?gemv reach them. Exits nonzero when any
// check fails.

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blis.h"
#include "cblas.h"

// The reference kernels of this configuration, which the SIMD kernels also
// call for the cases they do not handle.
DOTV_KER_PROT( float,  s, dotv_wasm32_ref )
DOTV_KER_PROT( double, d, dotv_wasm32_ref )
AXPYF_KER_PROT( float,  s, axpyf_wasm32_ref )
AXPYF_KER_PROT( double, d, axpyf_wasm32_ref )

// -- result accounting ----------------------------------------------------------

static unsigned long n_check = 0;
static unsigned long n_fail  = 0;

static void fail( const char* what, const char* where )
{
	if ( n_fail < 25 ) fprintf( stderr, "FAIL: %s (%s)\n", what, where );
	++n_fail;
}

#define CHECK( cond, what ) \
	do { ++n_check; if ( !( cond ) ) fail( what, label ); } while ( 0 )

// -- deterministic values --------------------------------------------------------

static uint32_t rng_state;

static void rng_seed( uint32_t s ) { rng_state = s ? s : 1u; }

static uint32_t rng_next( void )
{
	rng_state = rng_state * 1664525u + 1013904223u;
	return rng_state;
}

enum { EXACT, RANDOM, WITH_NAN, WITH_INF, NUM_MODES };

// EXACT values are multiples of 0.25 in [-2, 2]; RANDOM values lie in
// [-1, 1). The special modes use EXACT values plus one injected NaN or
// infinity.
static double value( int mode )
{
	if ( mode == RANDOM )
		return ( double )( rng_next() >> 8 ) / ( double )( 1u << 23 ) - 1.0;
	return ( double )( ( int )( ( rng_next() >> 8 ) % 17u ) - 8 ) * 0.25;
}

static double special( int mode, int which )
{
	if ( mode == WITH_NAN ) return NAN;
	return which % 2 ? -INFINITY : INFINITY;
}

// -- typed storage ----------------------------------------------------------------

typedef struct
{
	num_t        dt;
	const char*  name;
	size_t       size;
	double       eps;
	dotv_ker_ft  dotv, dotv_ref;
	axpyf_ker_ft axpyf, axpyf_ref;
	dim_t        long_m[3];   // around the axpyf four-column threshold
} type_t;

static double get( const type_t* t, const void* p, inc_t i )
{
	return t->dt == BLIS_FLOAT ? ( double )( ( const float* )p )[i]
	                           : ( ( const double* )p )[i];
}

static void put( const type_t* t, void* p, inc_t i, double v )
{
	if ( t->dt == BLIS_FLOAT ) ( ( float* )p )[i] = ( float )v;
	else                       ( ( double* )p )[i] = v;
}

// A strided vector of n elements in an allocation of exactly its extent,
// optionally preceded by one element so that the data are misaligned. With
// a negative increment the first element is at the highest address.
typedef struct
{
	unsigned char* alloc;
	void*          first;
	size_t         bytes;
} vec_t;

static size_t extent( dim_t n, inc_t inc )
{
	return n > 0 ? ( size_t )( n - 1 ) * ( size_t )( inc < 0 ? -inc : inc ) + 1 : 1;
}

static void vec_init( vec_t* v, const type_t* t, dim_t n, inc_t inc, int skew )
{
	const size_t count = extent( n, inc ) + ( skew ? 1 : 0 );
	v->bytes = count * t->size;
	v->alloc = malloc( v->bytes );
	if ( !v->alloc ) { fprintf( stderr, "out of memory\n" ); exit( 2 ); }
	// Elements that a strided access skips hold a recognizable pattern.
	for ( size_t i = 0; i < count; ++i ) put( t, v->alloc, ( inc_t )i, 1234.5 );
	unsigned char* start = v->alloc + ( skew ? t->size : 0 );
	v->first = start + ( inc < 0 && n > 0 ? ( size_t )( n - 1 ) * ( size_t )( -inc ) * t->size : 0 );
}

static int classes_match( double got, double want, double bound )
{
	if ( isnan( want ) ) return isnan( got );
	if ( isinf( want ) ) return got == want;
	return isfinite( got ) && fabs( got - want ) <= bound;
}

// -- dotv -----------------------------------------------------------------------

static void dotv_case( const type_t* t, dim_t n, inc_t incx, inc_t incy,
                       conj_t conjx, conj_t conjy, int mode, int skew )
{
	char label[160];
	snprintf( label, sizeof( label ), "%sdotv n=%ld incx=%ld incy=%ld conj=%d%d mode=%d skew=%d",
	          t->name, ( long )n, ( long )incx, ( long )incy,
	          conjx == BLIS_CONJUGATE, conjy == BLIS_CONJUGATE, mode, skew );

	vec_t x, y;
	vec_init( &x, t, n, incx, skew );
	vec_init( &y, t, n, incy, !skew );
	double sumabs = 0;
	for ( dim_t i = 0; i < n; ++i )
	{
		const double xi = value( mode ), yi = value( mode );
		put( t, x.first, i*incx, xi );
		put( t, y.first, i*incy, yi );
		sumabs += fabs( xi*yi );
	}
	if ( ( mode == WITH_NAN || mode == WITH_INF ) && n > 0 )
	{
		const dim_t at = ( dim_t )( rng_next() % ( uint32_t )n );
		if ( rng_next() % 2 ) put( t, x.first, at*incx, special( mode, ( int )at ) );
		else                  put( t, y.first, at*incy, special( mode, ( int )at ) );
	}
	unsigned char* x_saved = malloc( x.bytes );
	unsigned char* y_saved = malloc( y.bytes );
	memcpy( x_saved, x.alloc, x.bytes ); memcpy( y_saved, y.alloc, y.bytes );

	// The output lies between two guard elements and starts as a NaN, so an
	// unwritten result is also detected.
	double want_buf[3], got_buf[3];
	for ( int i = 0; i < 3; ++i ) { want_buf[i] = got_buf[i] = 0; }
	put( t, want_buf, 0, -7.0 ); put( t, want_buf, 1, NAN ); put( t, want_buf, 2, -9.0 );
	put( t, got_buf, 0, -7.0 );  put( t, got_buf, 1, NAN );  put( t, got_buf, 2, -9.0 );
	const cntx_t* cntx = bli_gks_query_cntx();
	t->dotv_ref( conjx, conjy, n, x.first, incx, y.first, incy,
	             ( unsigned char* )want_buf + t->size, cntx );
	t->dotv( conjx, conjy, n, x.first, incx, y.first, incy,
	         ( unsigned char* )got_buf + t->size, cntx );

	const double want = get( t, want_buf, 1 ), got = get( t, got_buf, 1 );
	const double bound = mode == RANDOM ? 2.0*( n + 2 )*t->eps*sumabs : 0.0;
	CHECK( classes_match( got, want, bound ), "dotv result differs from the reference kernel" );
	CHECK( n > 0 || ( got == 0.0 && want == 0.0 ), "dotv of length 0 is not zero" );
	CHECK( get( t, got_buf, 0 ) == -7.0 && get( t, got_buf, 2 ) == -9.0,
	       "dotv wrote outside its output" );
	CHECK( memcmp( x.alloc, x_saved, x.bytes ) == 0 && memcmp( y.alloc, y_saved, y.bytes ) == 0,
	       "dotv modified an input" );

	free( x_saved ); free( y_saved ); free( x.alloc ); free( y.alloc );
}

// -- axpyf ----------------------------------------------------------------------

enum { GUARD = 8 };

static void axpyf_case( const type_t* t, dim_t m, dim_t b_n, inc_t inca, inc_t lda_pad,
                        inc_t incx, inc_t incy, conj_t conja, conj_t conjx,
                        double alpha, int mode, int skew )
{
	char label[200];
	snprintf( label, sizeof( label ),
	          "%saxpyf m=%ld b_n=%ld inca=%ld lda+%ld incx=%ld incy=%ld conj=%d%d alpha=%g mode=%d skew=%d",
	          t->name, ( long )m, ( long )b_n, ( long )inca, ( long )lda_pad, ( long )incx,
	          ( long )incy, conja == BLIS_CONJUGATE, conjx == BLIS_CONJUGATE, alpha, mode, skew );

	// A is m x b_n with element (i, j) at i*inca + j*lda, allocated with
	// its exact extent.
	const inc_t lda = ( m > 0 ? ( m - 1 )*inca + 1 : 1 ) + lda_pad;
	const size_t a_count = ( m > 0 && b_n > 0 ) ? ( size_t )( ( m - 1 )*inca + ( b_n - 1 )*lda + 1 ) : 1;
	const size_t a_bytes = ( a_count + ( skew ? 1 : 0 ) )*t->size;
	unsigned char* a_alloc = malloc( a_bytes );
	if ( !a_alloc ) { fprintf( stderr, "out of memory\n" ); exit( 2 ); }
	memset( a_alloc, 0, a_bytes );
	void* a = a_alloc + ( skew ? t->size : 0 );

	vec_t x;
	vec_init( &x, t, b_n, incx, !skew );

	// y: the result is compared on a guarded copy and on a copy with its
	// exact extent.
	const size_t y_count = extent( m, incy ), y_skew = skew ? 1 : 0;
	const size_t y_bytes = ( GUARD + y_skew + y_count + GUARD )*t->size;
	unsigned char* y_init  = malloc( y_bytes );
	unsigned char* y_want  = malloc( y_bytes );
	unsigned char* y_got   = malloc( y_bytes );
	unsigned char* y_tight = malloc( ( y_count + y_skew )*t->size );
	if ( !y_init || !y_want || !y_got || !y_tight ) { fprintf( stderr, "out of memory\n" ); exit( 2 ); }
	for ( size_t i = 0; i < GUARD + y_skew + y_count + GUARD; ++i )
		put( t, y_init, ( inc_t )i, 4321.25 );
	const size_t y_first = GUARD + y_skew + ( incy < 0 && m > 0 ? ( size_t )( ( m - 1 )*( -incy ) ) : 0 );
	const size_t tight_first = y_skew + ( incy < 0 && m > 0 ? ( size_t )( ( m - 1 )*( -incy ) ) : 0 );

	double* bound = calloc( m > 0 ? ( size_t )m : 1, sizeof( double ) );
	for ( dim_t i = 0; i < m; ++i )
	{
		const double yi = value( mode );
		put( t, y_init, ( inc_t )y_first + i*incy, yi );
		bound[i] = fabs( yi );
	}
	double* ax = calloc( b_n > 0 ? ( size_t )b_n : 1, sizeof( double ) );
	for ( dim_t j = 0; j < b_n; ++j )
	{
		const double xj = value( mode );
		put( t, x.first, j*incx, xj );
		ax[j] = fabs( alpha*xj );
	}
	for ( dim_t j = 0; j < b_n; ++j )
	for ( dim_t i = 0; i < m; ++i )
	{
		const double aij = value( mode );
		put( t, a, i*inca + j*lda, aij );
		bound[i] += ax[j]*fabs( aij );
	}
	if ( ( mode == WITH_NAN || mode == WITH_INF ) && m > 0 && b_n > 0 )
	{
		const dim_t i = ( dim_t )( rng_next() % ( uint32_t )m );
		const dim_t j = ( dim_t )( rng_next() % ( uint32_t )b_n );
		switch ( rng_next() % 3 )
		{
			case 0:  put( t, a, i*inca + j*lda, special( mode, ( int )i ) ); break;
			case 1:  put( t, x.first, j*incx, special( mode, ( int )j ) ); break;
			default: put( t, y_init, ( inc_t )y_first + i*incy, special( mode, ( int )i ) ); break;
		}
	}
	memcpy( y_want, y_init, y_bytes );
	memcpy( y_got,  y_init, y_bytes );
	memcpy( y_tight, y_init + ( GUARD )*t->size, ( y_count + y_skew )*t->size );
	unsigned char* a_saved = malloc( a_bytes );
	unsigned char* x_saved = malloc( x.bytes );
	memcpy( a_saved, a_alloc, a_bytes ); memcpy( x_saved, x.alloc, x.bytes );

	double alpha_buf = 0; put( t, &alpha_buf, 0, alpha );
	const cntx_t* cntx = bli_gks_query_cntx();
	t->axpyf_ref( conja, conjx, m, b_n, &alpha_buf, a, inca, lda, x.first, incx,
	              y_want + y_first*t->size, incy, cntx );
	t->axpyf( conja, conjx, m, b_n, &alpha_buf, a, inca, lda, x.first, incx,
	          y_got + y_first*t->size, incy, cntx );
	t->axpyf( conja, conjx, m, b_n, &alpha_buf, a, inca, lda, x.first, incx,
	          y_tight + tight_first*t->size, incy, cntx );

	int ok = 1, tight_ok = 1;
	for ( dim_t i = 0; i < m; ++i )
	{
		const inc_t at = ( inc_t )y_first + i*incy;
		const double want = get( t, y_want, at ), got = get( t, y_got, at );
		const double b = mode == RANDOM ? 2.0*( b_n + 2 )*t->eps*bound[i] : 0.0;
		if ( !classes_match( got, want, b ) ) ok = 0;
		const double tight = get( t, y_tight, ( inc_t )tight_first + i*incy );
		if ( !( tight == got || ( isnan( tight ) && isnan( got ) ) ) ) tight_ok = 0;
	}
	CHECK( ok, "axpyf result differs from the reference kernel" );
	CHECK( tight_ok, "axpyf result depends on the memory around y" );

	// Every element of the guarded copy that is not an output must be intact.
	int guards_ok = 1;
	for ( size_t s = 0; s < GUARD + y_skew + y_count + GUARD; ++s )
	{
		const inc_t off = ( inc_t )s - ( inc_t )y_first;
		const int is_output = m > 0 && ( incy > 0 ? off >= 0 && off % incy == 0 && off / incy < m
		                                          : off <= 0 && off % incy == 0 && off / incy < m );
		if ( !is_output && memcmp( y_got + s*t->size, y_init + s*t->size, t->size ) ) guards_ok = 0;
	}
	CHECK( guards_ok, "axpyf wrote outside its output elements" );
	CHECK( memcmp( a_alloc, a_saved, a_bytes ) == 0 && memcmp( x.alloc, x_saved, x.bytes ) == 0,
	       "axpyf modified an input" );

	free( a_saved ); free( x_saved ); free( bound ); free( ax );
	free( y_init ); free( y_want ); free( y_got ); free( y_tight );
	free( a_alloc ); free( x.alloc );
}

// -- kernel contracts -------------------------------------------------------------

static void test_kernels( const type_t* t )
{
	static const inc_t incs[][2] = { { 1, 1 }, { 2, 1 }, { 1, 3 }, { -1, 1 }, { 2, -2 } };
	const int nincs = ( int )( sizeof( incs ) / sizeof( incs[0] ) );

	// dotv: lengths 0-67 and 1000, every increment pair, both conjugation
	// settings, every value mode, aligned and misaligned data.
	dim_t lens[72];
	int nlens = 0;
	for ( dim_t n = 0; n <= 67; ++n ) lens[nlens++] = n;
	lens[nlens++] = 1000;
	for ( int l = 0; l < nlens; ++l )
	for ( int s = 0; s < nincs; ++s )
	for ( int c = 0; c < 2; ++c )
	for ( int mode = 0; mode < NUM_MODES; ++mode )
		dotv_case( t, lens[l], incs[s][0], incs[s][1],
		           c ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
		           c ? BLIS_NO_CONJUGATE : BLIS_CONJUGATE, mode, ( l + s ) % 2 );

	// axpyf: the SIMD path needs unit increments and b_n == 8; the other
	// shapes check that the fallback to the reference kernel is exact.
	for ( int k = 0; k < 3; ++k ) lens[nlens++] = t->long_m[k];
	const double alphas[] = { 0.0, 1.0, -1.0, 0.75 };
	for ( int l = 0; l < nlens; ++l )
	for ( int mode = 0; mode < NUM_MODES; ++mode )
	for ( int a = 0; a < 4; ++a )
	{
		const dim_t m = lens[l];
		const double alpha = mode == RANDOM && a == 3 ? value( RANDOM ) : alphas[a];
		const int skew = ( l + a ) % 2;
		const conj_t conj = ( l + mode ) % 2 ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE;
		axpyf_case( t, m, 8, 1, a % 2 ? 3 : 0, 1, 1, conj, conj, alpha, mode, skew );
		if ( m > 70 && a > 0 ) continue;
		axpyf_case( t, m, 8, 2, 1, 1, 1, conj, conj, alpha, mode, skew );
		axpyf_case( t, m, 8, 1, 0, 2, 1, conj, conj, alpha, mode, skew );
		axpyf_case( t, m, 8, 1, 0, 1, -2, conj, conj, alpha, mode, skew );
		axpyf_case( t, m, 7, 1, 0, 1, 1, conj, conj, alpha, mode, skew );
		axpyf_case( t, m, 1, 1, 2, 1, 1, conj, conj, alpha, mode, skew );
		axpyf_case( t, m, 0, 1, 0, 1, 1, conj, conj, alpha, mode, skew );
	}
}

// -- public entry points ------------------------------------------------------------

static void_fp       saved_ker;
static unsigned long ker_calls;

static void count_dotv( conj_t cx, conj_t cy, dim_t n, const void* x, inc_t incx,
                        const void* y, inc_t incy, void* rho, const cntx_t* cntx )
{
	++ker_calls;
	( ( dotv_ker_ft )saved_ker )( cx, cy, n, x, incx, y, incy, rho, cntx );
}

static void count_axpyf( conj_t ca, conj_t cx, dim_t m, dim_t b_n, const void* alpha,
                         const void* a, inc_t inca, inc_t lda, const void* x, inc_t incx,
                         void* y, inc_t incy, const cntx_t* cntx )
{
	++ker_calls;
	( ( axpyf_ker_ft )saved_ker )( ca, cx, m, b_n, alpha, a, inca, lda, x, incx, y, incy, cntx );
}

static void test_public( const type_t* t )
{
	char label[64];
	snprintf( label, sizeof( label ), "%s public", t->name );
	cntx_t* cntx = ( cntx_t* )bli_gks_query_cntx();
	enum { M = 37, N = 19, LDA = 40 };
	double a[LDA*N], x[M], y[M], y0[M];
	float af[LDA*N], xf[M], yf[M];
	rng_seed( 11u );
	for ( int i = 0; i < LDA*N; ++i ) af[i] = ( float )( a[i] = value( EXACT ) );
	for ( int i = 0; i < M; ++i ) xf[i] = ( float )( x[i] = value( EXACT ) );
	for ( int i = 0; i < M; ++i ) yf[i] = ( float )( y[i] = y0[i] = value( EXACT ) );

	// cblas_?dot reaches the registered dotv kernel.
	saved_ker = bli_cntx_get_ukr_dt( t->dt, BLIS_DOTV_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )count_dotv, t->dt, BLIS_DOTV_KER, cntx );
	ker_calls = 0;
	double want = 0;
	for ( int i = 0; i < M; ++i ) want += x[i]*y[i];
	const double got = t->dt == BLIS_FLOAT ? ( double )cblas_sdot( M, xf, 1, yf, 1 )
	                                       : cblas_ddot( M, x, 1, y, 1 );
	bli_cntx_set_ukr_dt( saved_ker, t->dt, BLIS_DOTV_KER, cntx );
	CHECK( ker_calls > 0, "cblas dot did not reach the dotv kernel" );
	CHECK( got == want, "cblas dot result mismatch" );

	// cblas_?gemv without transposition reaches the registered axpyf kernel.
	saved_ker = bli_cntx_get_ukr_dt( t->dt, BLIS_AXPYF_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )count_axpyf, t->dt, BLIS_AXPYF_KER, cntx );
	ker_calls = 0;
	if ( t->dt == BLIS_FLOAT )
		cblas_sgemv( CblasColMajor, CblasNoTrans, M, N, 0.75f, af, LDA, xf, 1, -0.5f, yf, 1 );
	else
		cblas_dgemv( CblasColMajor, CblasNoTrans, M, N, 0.75, a, LDA, x, 1, -0.5, y, 1 );
	bli_cntx_set_ukr_dt( saved_ker, t->dt, BLIS_AXPYF_KER, cntx );
	CHECK( ker_calls > 0, "cblas gemv did not reach the axpyf kernel" );
	int ok = 1;
	for ( int i = 0; i < M; ++i )
	{
		double s = 0;
		for ( int j = 0; j < N; ++j ) s += a[i + j*LDA]*x[j];
		const double w = 0.75*s - 0.5*y0[i];
		const double g = t->dt == BLIS_FLOAT ? ( double )yf[i] : y[i];
		if ( g != w ) ok = 0;
	}
	CHECK( ok, "cblas gemv result mismatch" );
}

int main( void )
{
	bli_init();
	const cntx_t* cntx = bli_gks_query_cntx();
	const char* label = "registration";
	CHECK( bli_cntx_get_ukr_dt( BLIS_FLOAT,  BLIS_DOTV_KER, cntx ) ==
	       ( void_fp )bli_sdotv_wasm32_simd128 &&
	       bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_DOTV_KER, cntx ) ==
	       ( void_fp )bli_ddotv_wasm32_simd128,
	       "real dotv kernels are not the wasm32 simd128 kernels" );
	CHECK( bli_cntx_get_ukr_dt( BLIS_FLOAT,  BLIS_AXPYF_KER, cntx ) ==
	       ( void_fp )bli_saxpyf_wasm32_simd128_8 &&
	       bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPYF_KER, cntx ) ==
	       ( void_fp )bli_daxpyf_wasm32_simd128_8,
	       "real axpyf kernels are not the wasm32 simd128 kernels" );
	CHECK( bli_cntx_get_blksz_def_dt( BLIS_FLOAT,  BLIS_AF, cntx ) == 8 &&
	       bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_AF, cntx ) == 8,
	       "the axpyf fusing factor is not 8" );

	const type_t types[] =
	{
		{ BLIS_FLOAT, "s", sizeof( float ), FLT_EPSILON,
		  bli_sdotv_wasm32_simd128, bli_sdotv_wasm32_ref,
		  bli_saxpyf_wasm32_simd128_8, bli_saxpyf_wasm32_ref, { 1023, 1024, 1029 } },
		{ BLIS_DOUBLE, "d", sizeof( double ), DBL_EPSILON,
		  bli_ddotv_wasm32_simd128, bli_ddotv_wasm32_ref,
		  bli_daxpyf_wasm32_simd128_8, bli_daxpyf_wasm32_ref, { 511, 512, 515 } },
	};
	for ( int t = 0; t < 2; ++t )
	{
		rng_seed( 3u + ( uint32_t )t );
		test_kernels( &types[t] );
		test_public( &types[t] );
	}

	printf( "test-l1f: %lu checks, %lu failures\n", n_check, n_fail );
	bli_finalize();
	return n_fail ? 1 : 0;
}
