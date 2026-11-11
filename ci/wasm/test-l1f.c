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

// Contract test for the wasm32 real level-1v and level-1f kernels: dotv,
// axpyv, dotxv and axpyf. Every case runs the SIMD kernel and the reference
// kernel of this configuration on the same operands and compares the
// results:
//  - exactly, for inputs that are small multiples of 1/4, whose products and
//    sums are exact in both precisions at these lengths;
//  - within a rounding-error bound, for pseudo-random inputs;
//  - by class (NaN, +Inf or -Inf) when an input holds a NaN or an infinity.
// axpyf is also compared with a direct evaluation in double precision that
// applies every column in order, which the SIMD kernel does for every
// shape it takes (the reference kernel skips a column whose alpha * chi is
// zero for most shapes; see the kernel source).
// Inputs are allocated with their exact extent, so an AddressSanitizer build
// catches reads past the end, and must not change. Outputs are surrounded by
// guard elements, and the elements that a non-unit increment skips, which
// must not change either; each kernel under test also runs on a copy of its
// output with exactly its extent. Further sections check that the BLAS
// entry points reach the registered kernels and do not fall back to the
// unfused ones. Exits nonzero when any check fails.

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
AXPYV_KER_PROT( float,  s, axpyv_wasm32_ref )
AXPYV_KER_PROT( double, d, axpyv_wasm32_ref )
DOTXV_KER_PROT( float,  s, dotxv_wasm32_ref )
DOTXV_KER_PROT( double, d, dotxv_wasm32_ref )
AXPYF_KER_PROT( float,  s, axpyf_wasm32_ref )
AXPYF_KER_PROT( double, d, axpyf_wasm32_ref )

// -- result accounting ----------------------------------------------------------

static unsigned long n_check = 0;
static unsigned long n_fail  = 0;

// Prints the first three failures of each kind.
static void fail( const char* what, const char* where )
{
	static char kinds[64][96];
	static int  shown[64], n_kinds = 0;
	int k = 0;
	while ( k < n_kinds && strncmp( kinds[k], what, sizeof( kinds[k] ) - 1 ) ) ++k;
	if ( k == n_kinds && n_kinds < 64 )
	{
		snprintf( kinds[k], sizeof( kinds[k] ), "%s", what );
		shown[k] = 0;
		++n_kinds;
	}
	if ( k < 64 && shown[k]++ < 3 ) fprintf( stderr, "FAIL: %s (%s)\n", what, where );
	++n_fail;
}

#define CHECK( cond, what ) \
	do { ++n_check; if ( !( cond ) ) fail( what, label ); } while ( 0 )

static void* xmalloc( size_t bytes )
{
	void* p = malloc( bytes ? bytes : 1 );
	if ( !p ) { fprintf( stderr, "out of memory\n" ); exit( 2 ); }
	return p;
}

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

static int is_special( int mode ) { return mode == WITH_NAN || mode == WITH_INF; }

// -- typed storage ----------------------------------------------------------------

typedef struct
{
	num_t            dt;
	const char*      name;
	size_t           size;
	double           eps;
	dotv_ker_ft      dotv,      dotv_ref;
	axpyv_ker_ft     axpyv,     axpyv_ref;
	dotxv_ker_ft     dotxv,     dotxv_ref;
	axpyf_ker_ft     axpyf,     axpyf_ref;
	dim_t            long_m[3];   // around the axpyf four-column threshold
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

// Rounds v to the working precision of t.
static double rnd( const type_t* t, double v )
{
	return t->dt == BLIS_FLOAT ? ( double )( float )v : v;
}

static size_t extent( dim_t n, inc_t inc )
{
	return n > 0 ? ( size_t )( n - 1 ) * ( size_t )( inc < 0 ? -inc : inc ) + 1 : 1;
}

// Offset of element 0 from the lowest element: with a negative increment
// the first element is at the highest address.
static size_t first_offset( dim_t n, inc_t inc )
{
	return inc < 0 && n > 0 ? ( size_t )( n - 1 ) * ( size_t )( -inc ) : 0;
}

// An input vector of n elements in an allocation of exactly its extent,
// optionally preceded by one element so that the data are misaligned.
typedef struct
{
	unsigned char* alloc;
	void*          first;
	size_t         bytes;
} vec_t;

static void vec_init( vec_t* v, const type_t* t, dim_t n, inc_t inc, int skew )
{
	const size_t count = extent( n, inc ) + ( skew ? 1 : 0 );
	v->bytes = count * t->size;
	v->alloc = xmalloc( v->bytes );
	// Elements that a strided access skips hold a recognizable pattern.
	for ( size_t i = 0; i < count; ++i ) put( t, v->alloc, ( inc_t )i, 1234.5 );
	v->first = v->alloc + ( ( skew ? 1 : 0 ) + first_offset( n, inc ) ) * t->size;
}

// An m x n input matrix with element (i, j) at i*inca + j*lda, allocated
// with its exact extent. Unused elements hold the same pattern as above.
typedef struct
{
	unsigned char* alloc;
	void*          a;
	size_t         bytes;
	inc_t          inca, lda;
} mat_t;

static void mat_init( mat_t* A, const type_t* t, dim_t m, dim_t n, inc_t inca,
                      inc_t lda_pad, int skew )
{
	A->inca = inca;
	A->lda  = ( m > 0 ? ( m - 1 )*inca + 1 : 1 ) + lda_pad;
	const size_t count = ( m > 0 && n > 0 ) ? ( size_t )( ( m - 1 )*inca + ( n - 1 )*A->lda + 1 ) : 1;
	A->bytes = ( count + ( skew ? 1 : 0 ) )*t->size;
	A->alloc = xmalloc( A->bytes );
	for ( size_t i = 0; i < count + ( skew ? 1 : 0 ); ++i ) put( t, A->alloc, ( inc_t )i, 1234.5 );
	A->a = A->alloc + ( skew ? t->size : 0 );
}

static void mat_put( const type_t* t, mat_t* A, dim_t i, dim_t j, double v )
{
	put( t, A->a, i*A->inca + j*A->lda, v );
}

static double mat_get( const type_t* t, const mat_t* A, dim_t i, dim_t j )
{
	return get( t, A->a, i*A->inca + j*A->lda );
}

// An output (or updated) vector of n elements. It is kept in a guarded
// initial copy, a guarded copy for the kernel under test (got), a guarded
// copy for the reference kernel (want) and a copy with exactly its extent,
// which the kernel under test also updates (tight).
enum { GUARD = 8 };

typedef struct
{
	dim_t          n;
	inc_t          inc;
	size_t         skew, ext, count, first;
	unsigned char* init;
	unsigned char* got;
	unsigned char* want;
	unsigned char* tight;
} out_t;

static void out_init( out_t* o, const type_t* t, dim_t n, inc_t inc, int skew )
{
	o->n     = n;
	o->inc   = inc;
	o->skew  = skew ? 1 : 0;
	o->ext   = extent( n, inc );
	o->count = GUARD + o->skew + o->ext + GUARD;
	o->first = o->skew + first_offset( n, inc );
	o->init  = xmalloc( o->count*t->size );
	o->got   = xmalloc( o->count*t->size );
	o->want  = xmalloc( o->count*t->size );
	o->tight = xmalloc( ( o->skew + o->ext )*t->size );
	for ( size_t i = 0; i < o->count; ++i ) put( t, o->init, ( inc_t )i, 4321.25 );
}

static void out_set( const type_t* t, out_t* o, dim_t i, double v )
{
	put( t, o->init, ( inc_t )( GUARD + o->first ) + i*o->inc, v );
}

static double out_init_get( const type_t* t, const out_t* o, dim_t i )
{
	return get( t, o->init, ( inc_t )( GUARD + o->first ) + i*o->inc );
}

// Copies the initial contents to got, want and tight.
static void out_ready( const type_t* t, out_t* o )
{
	memcpy( o->got,  o->init, o->count*t->size );
	memcpy( o->want, o->init, o->count*t->size );
	memcpy( o->tight, o->init + GUARD*t->size, ( o->skew + o->ext )*t->size );
}

static void* out_ptr( const type_t* t, const out_t* o, unsigned char* buf )
{
	return buf == o->tight ? buf + o->first*t->size : buf + ( GUARD + o->first )*t->size;
}

static double out_get( const type_t* t, const out_t* o, const unsigned char* buf, dim_t i )
{
	const inc_t base = ( inc_t )( buf == o->tight ? o->first : GUARD + o->first );
	return get( t, buf, base + i*o->inc );
}

// Checks that every element of got that is not an output is intact, and
// that tight holds the same results as got.
static int out_guards_ok( const type_t* t, const out_t* o )
{
	for ( size_t s = 0; s < o->count; ++s )
	{
		const inc_t off = ( inc_t )s - ( inc_t )( GUARD + o->first );
		int is_output;
		if ( o->n == 0 )        is_output = 0;
		else if ( o->inc == 0 ) is_output = off == 0;
		else is_output = off % o->inc == 0 && off / o->inc >= 0 && off / o->inc < o->n;
		if ( !is_output && memcmp( o->got + s*t->size, o->init + s*t->size, t->size ) )
			return 0;
	}
	return 1;
}

static int out_tight_ok( const type_t* t, const out_t* o )
{
	for ( dim_t i = 0; i < o->n; ++i )
	{
		const double g = out_get( t, o, o->got, i ), h = out_get( t, o, o->tight, i );
		if ( !( g == h || ( isnan( g ) && isnan( h ) ) ) ) return 0;
	}
	return 1;
}

static void out_free( out_t* o )
{
	free( o->init ); free( o->got ); free( o->want ); free( o->tight );
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
	if ( is_special( mode ) && n > 0 )
	{
		const dim_t at = ( dim_t )( rng_next() % ( uint32_t )n );
		if ( rng_next() % 2 ) put( t, x.first, at*incx, special( mode, ( int )at ) );
		else                  put( t, y.first, at*incy, special( mode, ( int )at ) );
	}
	unsigned char* x_saved = xmalloc( x.bytes );
	unsigned char* y_saved = xmalloc( y.bytes );
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

// -- axpyv ----------------------------------------------------------------------

static void axpyv_case( const type_t* t, dim_t n, inc_t incx, inc_t incy,
                        conj_t conjx, double alpha, int mode, int skew )
{
	char label[160];
	snprintf( label, sizeof( label ), "%saxpyv n=%ld incx=%ld incy=%ld conj=%d alpha=%g mode=%d skew=%d",
	          t->name, ( long )n, ( long )incx, ( long )incy, conjx == BLIS_CONJUGATE,
	          alpha, mode, skew );

	vec_t x;
	out_t y;
	vec_init( &x, t, n, incx, skew );
	out_init( &y, t, n, incy, !skew );
	alpha = rnd( t, alpha );
	for ( dim_t i = 0; i < n; ++i )
	{
		put( t, x.first, i*incx, value( mode ) );
		out_set( t, &y, i, value( mode ) );
	}
	if ( is_special( mode ) && n > 0 )
	{
		const dim_t at = ( dim_t )( rng_next() % ( uint32_t )n );
		if ( rng_next() % 2 ) put( t, x.first, at*incx, special( mode, ( int )at ) );
		else                  out_set( t, &y, at, special( mode, ( int )at ) );
	}
	out_ready( t, &y );
	unsigned char* x_saved = xmalloc( x.bytes );
	memcpy( x_saved, x.alloc, x.bytes );

	double alpha_buf = 0; put( t, &alpha_buf, 0, alpha );
	const cntx_t* cntx = bli_gks_query_cntx();
	t->axpyv_ref( conjx, n, &alpha_buf, x.first, incx, out_ptr( t, &y, y.want ), incy, cntx );
	t->axpyv( conjx, n, &alpha_buf, x.first, incx, out_ptr( t, &y, y.got ), incy, cntx );
	t->axpyv( conjx, n, &alpha_buf, x.first, incx, out_ptr( t, &y, y.tight ), incy, cntx );

	int ok = 1;
	for ( dim_t i = 0; i < n; ++i )
	{
		const double xi = get( t, x.first, i*incx ), yi = out_init_get( t, &y, i );
		const double b = mode == RANDOM ? 4.0*t->eps*( fabs( yi ) + fabs( alpha*xi ) ) : 0.0;
		if ( !classes_match( out_get( t, &y, y.got, i ), out_get( t, &y, y.want, i ), b ) ) ok = 0;
	}
	CHECK( ok, "axpyv result differs from the reference kernel" );
	CHECK( out_tight_ok( t, &y ), "axpyv result depends on the memory around y" );
	CHECK( out_guards_ok( t, &y ), "axpyv wrote outside its output elements" );
	CHECK( memcmp( x.alloc, x_saved, x.bytes ) == 0, "axpyv modified an input" );

	free( x_saved ); free( x.alloc ); out_free( &y );
}

// -- dotxv ----------------------------------------------------------------------

static void dotxv_case( const type_t* t, dim_t n, inc_t incx, inc_t incy, conj_t conjx,
                        conj_t conjy, double alpha, double beta, int mode, int skew )
{
	char label[200];
	snprintf( label, sizeof( label ),
	          "%sdotxv n=%ld incx=%ld incy=%ld conj=%d%d alpha=%g beta=%g mode=%d skew=%d",
	          t->name, ( long )n, ( long )incx, ( long )incy, conjx == BLIS_CONJUGATE,
	          conjy == BLIS_CONJUGATE, alpha, beta, mode, skew );

	vec_t x, y;
	vec_init( &x, t, n, incx, skew );
	vec_init( &y, t, n, incy, !skew );
	alpha = rnd( t, alpha );
	beta  = rnd( t, beta );
	double sumabs = 0;
	for ( dim_t i = 0; i < n; ++i )
	{
		const double xi = value( mode ), yi = value( mode );
		put( t, x.first, i*incx, xi );
		put( t, y.first, i*incy, yi );
		sumabs += fabs( xi*yi );
	}
	// rho starts as a NaN when beta is zero, which must clear it.
	double rho0 = beta == 0.0 ? NAN : rnd( t, value( mode ) );
	if ( is_special( mode ) )
	{
		const dim_t at = n > 0 ? ( dim_t )( rng_next() % ( uint32_t )n ) : 0;
		switch ( n > 0 ? rng_next() % 3 : 2 )
		{
			case 0:  put( t, x.first, at*incx, special( mode, ( int )at ) ); break;
			case 1:  put( t, y.first, at*incy, special( mode, ( int )at ) ); break;
			default: rho0 = special( mode, ( int )at ); break;
		}
	}
	unsigned char* x_saved = xmalloc( x.bytes );
	unsigned char* y_saved = xmalloc( y.bytes );
	memcpy( x_saved, x.alloc, x.bytes ); memcpy( y_saved, y.alloc, y.bytes );

	double want_buf[3], got_buf[3];
	for ( int i = 0; i < 3; ++i ) { want_buf[i] = got_buf[i] = 0; }
	put( t, want_buf, 0, -7.0 ); put( t, want_buf, 1, rho0 ); put( t, want_buf, 2, -9.0 );
	put( t, got_buf, 0, -7.0 );  put( t, got_buf, 1, rho0 );  put( t, got_buf, 2, -9.0 );
	double alpha_buf = 0, beta_buf = 0;
	put( t, &alpha_buf, 0, alpha ); put( t, &beta_buf, 0, beta );
	const cntx_t* cntx = bli_gks_query_cntx();
	t->dotxv_ref( conjx, conjy, n, &alpha_buf, x.first, incx, y.first, incy, &beta_buf,
	              ( unsigned char* )want_buf + t->size, cntx );
	t->dotxv( conjx, conjy, n, &alpha_buf, x.first, incx, y.first, incy, &beta_buf,
	          ( unsigned char* )got_buf + t->size, cntx );

	const double want = get( t, want_buf, 1 ), got = get( t, got_buf, 1 );
	const double bound = mode == RANDOM
	                     ? 2.0*( n + 3 )*t->eps*( ( beta == 0.0 ? 0.0 : fabs( beta*rho0 ) ) +
	                                              fabs( alpha )*sumabs ) : 0.0;
	CHECK( classes_match( got, want, bound ), "dotxv result differs from the reference kernel" );
	CHECK( get( t, got_buf, 0 ) == -7.0 && get( t, got_buf, 2 ) == -9.0,
	       "dotxv wrote outside its output" );
	CHECK( memcmp( x.alloc, x_saved, x.bytes ) == 0 && memcmp( y.alloc, y_saved, y.bytes ) == 0,
	       "dotxv modified an input" );

	free( x_saved ); free( y_saved ); free( x.alloc ); free( y.alloc );
}

// -- axpyf ----------------------------------------------------------------------

static void axpyf_case( const type_t* t, dim_t m, dim_t b_n, inc_t inca, inc_t lda_pad,
                        inc_t incx, inc_t incy, conj_t conja, conj_t conjx,
                        double alpha, int mode, int skew )
{
	char label[200];
	snprintf( label, sizeof( label ),
	          "%saxpyf m=%ld b_n=%ld inca=%ld lda+%ld incx=%ld incy=%ld conj=%d%d alpha=%g mode=%d skew=%d",
	          t->name, ( long )m, ( long )b_n, ( long )inca, ( long )lda_pad, ( long )incx,
	          ( long )incy, conja == BLIS_CONJUGATE, conjx == BLIS_CONJUGATE, alpha, mode, skew );

	mat_t A;
	vec_t x;
	out_t y;
	mat_init( &A, t, m, b_n, inca, lda_pad, skew );
	vec_init( &x, t, b_n, incx, !skew );
	out_init( &y, t, m, incy, skew );
	alpha = rnd( t, alpha );

	for ( dim_t i = 0; i < m; ++i ) out_set( t, &y, i, value( mode ) );
	for ( dim_t j = 0; j < b_n; ++j ) put( t, x.first, j*incx, value( mode ) );
	for ( dim_t j = 0; j < b_n; ++j )
	for ( dim_t i = 0; i < m; ++i ) mat_put( t, &A, i, j, value( mode ) );
	if ( is_special( mode ) && m > 0 && b_n > 0 )
	{
		const dim_t i = ( dim_t )( rng_next() % ( uint32_t )m );
		const dim_t j = ( dim_t )( rng_next() % ( uint32_t )b_n );
		switch ( rng_next() % 3 )
		{
			case 0:  mat_put( t, &A, i, j, special( mode, ( int )i ) ); break;
			case 1:  put( t, x.first, j*incx, special( mode, ( int )j ) ); break;
			default: out_set( t, &y, i, special( mode, ( int )i ) ); break;
		}
	}
	out_ready( t, &y );
	unsigned char* a_saved = xmalloc( A.bytes );
	unsigned char* x_saved = xmalloc( x.bytes );
	memcpy( a_saved, A.alloc, A.bytes ); memcpy( x_saved, x.alloc, x.bytes );

	double alpha_buf = 0; put( t, &alpha_buf, 0, alpha );
	const cntx_t* cntx = bli_gks_query_cntx();
	t->axpyf_ref( conja, conjx, m, b_n, &alpha_buf, A.a, inca, A.lda, x.first, incx,
	              out_ptr( t, &y, y.want ), incy, cntx );
	t->axpyf( conja, conjx, m, b_n, &alpha_buf, A.a, inca, A.lda, x.first, incx,
	          out_ptr( t, &y, y.got ), incy, cntx );
	t->axpyf( conja, conjx, m, b_n, &alpha_buf, A.a, inca, A.lda, x.first, incx,
	          out_ptr( t, &y, y.tight ), incy, cntx );

	// The SIMD path applies every column in order. The reference kernel does
	// so only with unit increments and b_n == 8, so special values are
	// compared with it only then (and for the shapes passed to it). Without
	// special values both agree with the direct evaluation for every shape
	// except incy == 0, where every output is the same element.
	const int simd_path = inca == 1 && incy != 0;
	const int ref_same  = !is_special( mode ) || !simd_path ||
	                      ( incx == 1 && incy == 1 && b_n == 8 );
	int ok_ref = 1, ok_direct = 1;
	for ( dim_t i = 0; i < m; ++i )
	{
		const double got = out_get( t, &y, y.got, i );
		double direct = out_init_get( t, &y, i ), sumabs = fabs( direct );
		for ( dim_t j = 0; j < b_n; ++j )
		{
			const double axj = rnd( t, alpha*get( t, x.first, j*incx ) );
			direct += axj*mat_get( t, &A, i, j );
			sumabs += fabs( axj*mat_get( t, &A, i, j ) );
		}
		const double b = mode == RANDOM ? 2.0*( b_n + 2 )*t->eps*sumabs : 0.0;
		if ( ref_same && !classes_match( got, out_get( t, &y, y.want, i ), b ) ) ok_ref = 0;
		if ( ( simd_path || ( incy != 0 && !is_special( mode ) ) ) &&
		     !classes_match( got, direct, b ) ) ok_direct = 0;
	}
	CHECK( ok_ref, "axpyf result differs from the reference kernel" );
	CHECK( ok_direct, "axpyf result differs from the direct evaluation" );
	CHECK( out_tight_ok( t, &y ), "axpyf result depends on the memory around y" );
	CHECK( out_guards_ok( t, &y ), "axpyf wrote outside its output elements" );
	CHECK( memcmp( A.alloc, a_saved, A.bytes ) == 0 && memcmp( x.alloc, x_saved, x.bytes ) == 0,
	       "axpyf modified an input" );

	free( a_saved ); free( x_saved ); free( A.alloc ); free( x.alloc ); out_free( &y );
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

	// axpyv and dotxv: the same lengths and increments; alpha (and beta)
	// include the values the kernels treat separately.
	const double alphas[] = { 0.0, 1.0, -1.0, 0.75 };
	const double betas[]  = { 0.0, 1.0, -0.5, 0.0 };
	for ( int l = 0; l < nlens; ++l )
	for ( int s = 0; s < nincs; ++s )
	for ( int mode = 0; mode < NUM_MODES; ++mode )
	for ( int a = 0; a < 4; ++a )
	{
		const double alpha = mode == RANDOM && a == 3 ? value( RANDOM ) : alphas[a];
		const conj_t conj  = ( l + a ) % 2 ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE;
		const int    skew  = ( l + s + a ) % 2;
		axpyv_case( t, lens[l], incs[s][0], incs[s][1], conj, alpha, mode, skew );
		dotxv_case( t, lens[l], incs[s][0], incs[s][1], conj, BLIS_NO_CONJUGATE,
		            alpha, betas[( a + s ) % 4], mode, skew );
	}

	// axpyf: the SIMD path needs contiguous columns and a nonzero incy; it
	// takes any b_n in passes of eight, four, two and one columns. The
	// other shapes check that the fallback to the reference kernel is exact.
	for ( int k = 0; k < 3; ++k ) lens[nlens++] = t->long_m[k];
	static const dim_t axpyf_bn[] = { 7, 6, 5, 4, 3, 2, 1, 0, 9, 12, 13, 16, 17 };
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
		axpyf_case( t, m, 8, 1, 0, 2, 1, conj, conj, alpha, mode, skew );
		axpyf_case( t, m, 8, 1, 1, -1, 3, conj, conj, alpha, mode, skew );
		axpyf_case( t, m, 8, 1, 0, 1, -2, conj, conj, alpha, mode, skew );
		axpyf_case( t, m, axpyf_bn[( l + a ) % 13], 1, a, 1, 1, conj, conj, alpha, mode, skew );
		axpyf_case( t, m, axpyf_bn[( l + a + 5 ) % 13], 1, 0, 3, -1, conj, conj, alpha, mode, skew );
		if ( m > 40 ) continue;
		axpyf_case( t, m, 8, 2, 1, 1, 1, conj, conj, alpha, mode, skew );
		axpyf_case( t, m, 3, 1, 0, 1, 0, conj, conj, alpha, mode, skew );
		for ( int k = 0; k < 13; ++k )
			axpyf_case( t, m, axpyf_bn[k], 1, 0, 1, 1, conj, conj, alpha, mode, skew );
	}

}

// -- public entry points ------------------------------------------------------------

// Counting wrappers for the registered kernels; each forwards to the kernel
// it replaced.
static dotv_ker_ft      saved_dotv;
static axpyv_ker_ft     saved_axpyv;
static dotxv_ker_ft     saved_dotxv;
static axpyf_ker_ft     saved_axpyf;
static dotxf_ker_ft     saved_dotxf;
static dotxaxpyf_ker_ft saved_dotxaxpyf;
static unsigned long    calls_dotv, calls_axpyv, calls_dotxv;
static unsigned long    calls_axpyf, calls_dotxf, calls_dotxaxpyf;

static void count_dotv( dotv_params, const cntx_t* cntx )
{
	++calls_dotv;
	saved_dotv( conjx, conjy, n, x, incx, y, incy, rho, cntx );
}

static void count_axpyv( axpyv_params, const cntx_t* cntx )
{
	++calls_axpyv;
	saved_axpyv( conjx, n, alpha, x, incx, y, incy, cntx );
}

static void count_dotxv( dotxv_params, const cntx_t* cntx )
{
	++calls_dotxv;
	saved_dotxv( conjx, conjy, n, alpha, x, incx, y, incy, beta, rho, cntx );
}

static void count_axpyf( axpyf_params, const cntx_t* cntx )
{
	++calls_axpyf;
	saved_axpyf( conja, conjx, m, b_n, alpha, a, inca, lda, x, incx, y, incy, cntx );
}

static void count_dotxf( dotxf_params, const cntx_t* cntx )
{
	++calls_dotxf;
	saved_dotxf( conjat, conjx, m, b_n, alpha, a, inca, lda, x, incx, beta, y, incy, cntx );
}

static void count_dotxaxpyf( dotxaxpyf_params, const cntx_t* cntx )
{
	++calls_dotxaxpyf;
	saved_dotxaxpyf( conjat, conja, conjw, conjx, m, b_n, alpha, a, inca, lda, w, incw,
	                 x, incx, beta, y, incy, z, incz, cntx );
}

// Installs the counting wrappers for dt in the global context and clears
// the counts; uninstall() puts the registered kernels back.
static void install( num_t dt )
{
	cntx_t* cntx = ( cntx_t* )bli_gks_query_cntx();
	saved_dotv      = bli_cntx_get_ukr_dt( dt, BLIS_DOTV_KER, cntx );
	saved_axpyv     = bli_cntx_get_ukr_dt( dt, BLIS_AXPYV_KER, cntx );
	saved_dotxv     = bli_cntx_get_ukr_dt( dt, BLIS_DOTXV_KER, cntx );
	saved_axpyf     = bli_cntx_get_ukr_dt( dt, BLIS_AXPYF_KER, cntx );
	saved_dotxf     = bli_cntx_get_ukr_dt( dt, BLIS_DOTXF_KER, cntx );
	saved_dotxaxpyf = bli_cntx_get_ukr_dt( dt, BLIS_DOTXAXPYF_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )count_dotv,      dt, BLIS_DOTV_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )count_axpyv,     dt, BLIS_AXPYV_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )count_dotxv,     dt, BLIS_DOTXV_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )count_axpyf,     dt, BLIS_AXPYF_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )count_dotxf,     dt, BLIS_DOTXF_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )count_dotxaxpyf, dt, BLIS_DOTXAXPYF_KER, cntx );
	calls_dotv = calls_axpyv = calls_dotxv = 0;
	calls_axpyf = calls_dotxf = calls_dotxaxpyf = 0;
}

static void uninstall( num_t dt )
{
	cntx_t* cntx = ( cntx_t* )bli_gks_query_cntx();
	bli_cntx_set_ukr_dt( ( void_fp )saved_dotv,      dt, BLIS_DOTV_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )saved_axpyv,     dt, BLIS_AXPYV_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )saved_dotxv,     dt, BLIS_DOTXV_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )saved_axpyf,     dt, BLIS_AXPYF_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )saved_dotxf,     dt, BLIS_DOTXF_KER, cntx );
	bli_cntx_set_ukr_dt( ( void_fp )saved_dotxaxpyf, dt, BLIS_DOTXAXPYF_KER, cntx );
}

enum { M = 37, N = 19, LDA = 40, VMAX = 3*M };

// Operands of the public calls: exact values, kept in double and in float.
static double a_d[LDA*M], x_d[VMAX], y_d[VMAX], y0_d[VMAX];
static float  a_s[LDA*M], x_s[VMAX], y_s[VMAX];

static void fill_public( void )
{
	for ( int i = 0; i < LDA*M; ++i ) a_s[i] = ( float )( a_d[i] = value( EXACT ) );
	for ( int i = 0; i < VMAX; ++i ) x_s[i] = ( float )( x_d[i] = value( EXACT ) );
	for ( int i = 0; i < VMAX; ++i ) y_s[i] = ( float )( y_d[i] = y0_d[i] = value( EXACT ) );
}

static double y_at( const type_t* t, int i )
{
	return t->dt == BLIS_FLOAT ? ( double )y_s[i] : y_d[i];
}

// Checks y against y0 updated by alpha * op(A) * x + (beta - 1) * y0 at
// the given increments, and that the elements between them are unchanged.
// A is m x n with leading dimension LDA; op(A) is A or A^T.
static int gemv_ok( const type_t* t, int trans, int m, int n, double alpha,
                    int incx, double beta, int incy )
{
	const int ny = trans ? n : m, nx = trans ? m : n;
	for ( int k = 0; k < ny*incy; ++k )
	{
		double want = y0_d[k];
		if ( k % incy == 0 )
		{
			const int i = k / incy;
			double s = 0;
			for ( int j = 0; j < nx; ++j )
				s += ( trans ? a_d[j + i*LDA] : a_d[i + j*LDA] )*x_d[j*incx];
			want = alpha*s + beta*y0_d[k];
		}
		if ( y_at( t, k ) != want ) return 0;
	}
	return 1;
}

static void test_public( const type_t* t )
{
	char label[96];
	const num_t dt = t->dt;
	const int   is_s = dt == BLIS_FLOAT;
	rng_seed( 11u );
	fill_public();

	// cblas_?dot reaches the registered dotv kernel.
	snprintf( label, sizeof( label ), "%s public dot", t->name );
	install( dt );
	double want = 0;
	for ( int i = 0; i < M; ++i ) want += x_d[i]*y_d[i];
	const double got = is_s ? ( double )cblas_sdot( M, x_s, 1, y_s, 1 )
	                        : cblas_ddot( M, x_d, 1, y_d, 1 );
	uninstall( dt );
	CHECK( calls_dotv > 0, "cblas dot did not reach the dotv kernel" );
	CHECK( got == want, "cblas dot result mismatch" );

	// cblas_?gemv without transposition reaches the registered axpyf kernel
	// for every increment of x and y and never falls back to axpyv.
	static const int gemv_inc[][2] = { { 1, 1 }, { 2, 1 }, { 1, 3 }, { 3, 2 } };
	for ( int k = 0; k < 4; ++k )
	{
		const int incx = gemv_inc[k][0], incy = gemv_inc[k][1];
		snprintf( label, sizeof( label ), "%s public gemv n incx=%d incy=%d", t->name, incx, incy );
		fill_public();
		install( dt );
		if ( is_s )
			cblas_sgemv( CblasColMajor, CblasNoTrans, M, N, 0.75f, a_s, LDA, x_s, incx, -0.5f, y_s, incy );
		else
			cblas_dgemv( CblasColMajor, CblasNoTrans, M, N, 0.75, a_d, LDA, x_d, incx, -0.5, y_d, incy );
		uninstall( dt );
		CHECK( calls_axpyf > 0, "cblas gemv did not reach the axpyf kernel" );
		CHECK( calls_axpyv == 0, "cblas gemv fell back to the axpyv kernel" );
		CHECK( gemv_ok( t, 0, M, N, 0.75, incx, -0.5, incy ), "cblas gemv result mismatch" );
	}

	// cblas_?ger reaches axpyv once per column.
	snprintf( label, sizeof( label ), "%s public ger", t->name );
	fill_public();
	double a0[LDA*N];
	memcpy( a0, a_d, sizeof( a0 ) );
	install( dt );
	if ( is_s ) cblas_sger( CblasColMajor, M, N, -0.75f, x_s, 1, y_s, 2, a_s, LDA );
	else        cblas_dger( CblasColMajor, M, N, -0.75, x_d, 1, y_d, 2, a_d, LDA );
	uninstall( dt );
	CHECK( calls_axpyv == N, "cblas ger did not call axpyv once per column" );
	int ok = 1;
	for ( int j = 0; j < N; ++j )
	for ( int i = 0; i < LDA; ++i )
	{
		const double w = i < M ? a0[i + j*LDA] - 0.75*x_d[i]*y0_d[2*j] : a0[i + j*LDA];
		const double g = is_s ? ( double )a_s[i + j*LDA] : a_d[i + j*LDA];
		if ( g != w ) ok = 0;
	}
	CHECK( ok, "cblas ger result mismatch" );

}

int main( void )
{
	bli_init();
	const cntx_t* cntx = bli_gks_query_cntx();
	const char* label = "registration";
	static const struct { kerid_t id; void_fp s, d; const char* name; } regs[] =
	{
		{ BLIS_DOTV_KER,      ( void_fp )bli_sdotv_wasm32_simd128,      ( void_fp )bli_ddotv_wasm32_simd128,      "dotv" },
		{ BLIS_AXPYV_KER,     ( void_fp )bli_saxpyv_wasm32_simd128,     ( void_fp )bli_daxpyv_wasm32_simd128,     "axpyv" },
		{ BLIS_DOTXV_KER,     ( void_fp )bli_sdotxv_wasm32_simd128,     ( void_fp )bli_ddotxv_wasm32_simd128,     "dotxv" },
		{ BLIS_AXPYF_KER,     ( void_fp )bli_saxpyf_wasm32_simd128_8,   ( void_fp )bli_daxpyf_wasm32_simd128_8,   "axpyf" },
	};
	for ( size_t r = 0; r < sizeof( regs )/sizeof( regs[0] ); ++r )
	{
		char what[96];
		snprintf( what, sizeof( what ), "real %s kernels are not the wasm32 simd128 kernels", regs[r].name );
		CHECK( bli_cntx_get_ukr_dt( BLIS_FLOAT,  regs[r].id, cntx ) == regs[r].s &&
		       bli_cntx_get_ukr_dt( BLIS_DOUBLE, regs[r].id, cntx ) == regs[r].d, what );
	}
	CHECK( bli_cntx_get_blksz_def_dt( BLIS_FLOAT,  BLIS_AF, cntx ) == 8 &&
	       bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_AF, cntx ) == 8,
	       "the axpyf fusing factor is not 8" );

	const type_t types[] =
	{
		{ BLIS_FLOAT, "s", sizeof( float ), FLT_EPSILON,
		  bli_sdotv_wasm32_simd128,      bli_sdotv_wasm32_ref,
		  bli_saxpyv_wasm32_simd128,     bli_saxpyv_wasm32_ref,
		  bli_sdotxv_wasm32_simd128,     bli_sdotxv_wasm32_ref,
		  bli_saxpyf_wasm32_simd128_8,   bli_saxpyf_wasm32_ref, { 1023, 1024, 1029 } },
		{ BLIS_DOUBLE, "d", sizeof( double ), DBL_EPSILON,
		  bli_ddotv_wasm32_simd128,      bli_ddotv_wasm32_ref,
		  bli_daxpyv_wasm32_simd128,     bli_daxpyv_wasm32_ref,
		  bli_ddotxv_wasm32_simd128,     bli_ddotxv_wasm32_ref,
		  bli_daxpyf_wasm32_simd128_8,   bli_daxpyf_wasm32_ref, { 511, 512, 515 } },
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
