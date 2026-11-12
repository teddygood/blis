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

// Contract test for the small single-precision GEMM handler that the wasm32
// context registers for BLIS_GEMM (bli_sgemmsup_wasm32_simd128).
//
// Every case makes a public call (CBLAS or the object API) and checks the
// result against an oracle evaluated in double precision, together with the
// padding and guard elements around every operand. The same problem is then
// repeated on fresh operands through the small-problem stage alone: the
// handler itself for the CBLAS cases, and bli_gemmsup()/bli_gemmtsup(), which
// bli_gemm_ex()/bli_gemmt_ex() run first, for the object cases. That stage
// must take exactly the eligible problems and compute them correctly, and
// must leave every operand untouched when it declines. Fallback cases replace
// the context's kernels or thresholds to check that the handler declines.
//
// Padding only reveals a read outside an operand when the value read reaches
// a result. A subset of the CBLAS cases therefore allocates A and B with
// their exact extent: tight leading dimensions and no offset, padding or
// guard elements. In the AddressSanitizer build, which instruments the
// handler, a read before the first or past the last element of such an
// operand fails the test.
//
// A build configured with --disable-sup-handling defines
// BLIS_DISABLE_SUP_HANDLING, and then no problem may take the small path.
//
// Compiled with -DTEST_SMALL_STACK, for a run with a deliberately small
// stack, the test skips the public calls that BLIS computes on its
// conventional path: that path keeps temporary microtiles on the stack and
// needs more than the small path. Every call of the small-problem stage
// still runs.

#include "blis.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t custom_gemm_hits, custom_pack_hits;
static uint64_t cases, values, accepted, declined;
static cntx_t context;
static void_fp original_gemm, original_pack;

typedef struct { double r, i; } value_t;
typedef struct
{
	obj_t obj;
	num_t dt;
	dim_t m, n;
	inc_t rs, cs;
	int trans, offm, offn;
	size_t count, origin, elem;
	unsigned char* data;
	unsigned char* saved;
	unsigned char* active;
} matrix_t;

typedef struct
{
	matrix_t a, b, c;
	obj_t alpha, beta;
	float af, bf;
	double ad, bd;
} fixture_t;

static value_t add( value_t a, value_t b )
{
	return ( value_t ){ a.r + b.r, a.i + b.i };
}
static value_t mul( value_t a, value_t b )
{
	return ( value_t ){ a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r };
}
static double mag( value_t a ) { return fabs( a.r ) + fabs( a.i ); }
static int is_trans( int t ) { return t == 1 || t == 2; }
static int is_conj( int t ) { return t == 2 || t == 3; }
static trans_t trans_code( int t )
{
	const trans_t codes[] = { BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE,
	                         BLIS_CONJ_TRANSPOSE, BLIS_CONJ_NO_TRANSPOSE };
	return codes[t];
}
static size_t index_at( const matrix_t* x, dim_t i, dim_t j )
{
	const dim_t pi = is_trans( x->trans ) ? j : i;
	const dim_t pj = is_trans( x->trans ) ? i : j;
	return ( size_t )( ( inc_t )x->origin +
	                  ( pi + x->offm )*x->rs + ( pj + x->offn )*x->cs );
}
static value_t load( const matrix_t* x, dim_t i, dim_t j, int saved )
{
	const unsigned char* p = ( saved ? x->saved : x->data ) + index_at( x, i, j )*x->elem;
	value_t v = { 0, 0 };
	if ( x->dt == BLIS_FLOAT ) { float z; memcpy( &z, p, sizeof z ); v.r = z; }
	else if ( x->dt == BLIS_DOUBLE ) { double z; memcpy( &z, p, sizeof z ); v.r = z; }
	else
	{
		scomplex z;
		memcpy( &z, p, sizeof z );
		v.r = bli_creal( z ); v.i = bli_cimag( z );
	}
	if ( is_conj( x->trans ) ) v.i = -v.i;
	return v;
}
static void store_at( matrix_t* x, dim_t i, dim_t j, value_t v )
{
	const size_t at = index_at( x, i, j );
	unsigned char* p = x->data + at*x->elem;
	if ( x->dt == BLIS_FLOAT ) { const float z = ( float )v.r; memcpy( p, &z, sizeof z ); }
	else if ( x->dt == BLIS_DOUBLE ) { memcpy( p, &v.r, sizeof v.r ); }
	else
	{
		scomplex z;
		bli_tsets( c, c, v.r, v.i, z );
		memcpy( p, &z, sizeof z );
	}
	x->active[at] = 1;
}

// Layouts 0/1 are positive row/column storage. Layout 2 has two nonunit
// strides. Layouts 3/4/5 reverse the row, column, or both strides. All are
// nonoverlapping. Matrix offsets leave untouched storage before each view.
// An exact matrix (layout 0 or 1, no offset, nonzero dimensions) has tight
// strides and no padding or guards, so its allocation ends at its last
// element.
static int matrix_init
     (
       matrix_t* x, num_t dt, dim_t m, dim_t n, int layout,
       int trans, int offset, int seed, int poison, int exact
     )
{
	memset( x, 0, sizeof *x );
	x->dt = dt; x->m = m; x->n = n; x->trans = trans;
	x->offm = offset ? 2 : 0; x->offn = offset ? 1 : 0;
	const dim_t pm = is_trans( trans ) ? n : m;
	const dim_t pn = is_trans( trans ) ? m : n;
	x->rs = pn + x->offn + ( exact ? 0 : 5 ); x->cs = 1;
	if ( layout == 1 ) { x->rs = 1; x->cs = pm + x->offm + ( exact ? 0 : 7 ); }
	if ( layout == 2 ) { x->cs = 2; x->rs = 2*( pn + x->offn ) + 5; }
	if ( layout == 3 || layout == 5 ) x->rs = -x->rs;
	if ( layout == 4 || layout == 5 ) x->cs = -x->cs;
	const inc_t hi_i = ( pm > 0 ? pm - 1 : 0 ) + x->offm;
	const inc_t hi_j = ( pn > 0 ? pn - 1 : 0 ) + x->offn;
	const inc_t low = ( x->rs < 0 ? hi_i*x->rs : 0 ) +
	                  ( x->cs < 0 ? hi_j*x->cs : 0 );
	const inc_t high = ( x->rs > 0 ? hi_i*x->rs : 0 ) +
	                   ( x->cs > 0 ? hi_j*x->cs : 0 );
	x->origin = ( size_t )( ( exact ? 0 : 17 ) - low );
	x->count = x->origin + ( size_t )high + ( exact ? 1 : 19 );
	x->elem = dt == BLIS_FLOAT ? sizeof( float ) :
	          dt == BLIS_DOUBLE ? sizeof( double ) : sizeof( scomplex );
	x->data = malloc( x->count*x->elem );
	x->saved = malloc( x->count*x->elem );
	x->active = calloc( x->count, 1 );
	if ( !x->data || !x->saved || !x->active ) return 1;
	// The padding and guards hold signaling NaNs. A padding read that
	// contributes to a result makes it NaN, and arithmetic returns a quiet
	// NaN with different bits, so a padding write computed from padding
	// changes the stored bytes too.
	const uint64_t snan_d = 0x7ff4a5a5a5a5a5a5ULL;
	const uint32_t snan_s = 0x7fa5a5a5u;
	for ( size_t p = 0; p < x->count; ++p )
	{
		unsigned char* e = x->data + p*x->elem;
		if ( dt == BLIS_DOUBLE ) memcpy( e, &snan_d, sizeof snan_d );
		else
			for ( size_t w = 0; w < x->elem; w += sizeof snan_s )
				memcpy( e + w, &snan_s, sizeof snan_s );
	}
	x->obj = ( obj_t )BLIS_OBJECT_INITIALIZER;
	bli_obj_init_finish( dt, pm, pn, x->data + x->origin*x->elem,
	                     x->rs, x->cs, &x->obj );
	bli_obj_set_offs( x->offm, x->offn, &x->obj );
	bli_obj_set_conjtrans( trans_code( trans ), &x->obj );
	for ( dim_t i = 0; i < m; ++i )
	for ( dim_t j = 0; j < n; ++j )
	{
		value_t v = { ( ( i*13 + j*7 + seed*11 ) % 61 - 30 ) / 17.0,
		              ( ( i*5 + j*19 + seed*3 ) % 31 - 15 ) / 23.0 };
		if ( dt != BLIS_SCOMPLEX ) v.i = 0;
		if ( poison ) v.r = ( i + j ) % 3 == 0 ? NAN :
		                         ( ( i + j ) % 3 == 1 ? INFINITY : -INFINITY );
		store_at( x, i, j, v );
	}
	memcpy( x->saved, x->data, x->count*x->elem );
	return 0;
}
static void matrix_free( matrix_t* x )
{
	free( x->data ); free( x->saved ); free( x->active );
}
static void fixture_free( fixture_t* f )
{
	matrix_free( &f->a ); matrix_free( &f->b ); matrix_free( &f->c );
}
static int fixture_init
     (
       fixture_t* f, dim_t m, dim_t n, dim_t k,
       int la, int lb, int lc, int ta, int tb, int tc,
       num_t da, num_t db, num_t dc, float alpha, float beta, int offset, int cast,
       int exact
     )
{
	memset( f, 0, sizeof *f );
	f->af = alpha; f->bf = beta;
	f->ad = ( double )alpha + ( cast ? 1e-10 : 0 );
	f->bd = ( double )beta + ( cast && beta != 0.0f ? 1e-10 : 0 );
	// Only A and B can be exact: the guards around C detect stray writes.
	if ( matrix_init( &f->a, da, m, k, la, ta, offset, 1, alpha == 0.0f, exact ) ||
	     matrix_init( &f->b, db, k, n, lb, tb, offset, 2, alpha == 0.0f, exact ) ||
	     matrix_init( &f->c, dc, m, n, lc, tc, offset, 3, beta == 0.0f, 0 ) )
	{
		fixture_free( f ); return 1;
	}
	f->alpha = ( obj_t )BLIS_OBJECT_INITIALIZER;
	f->beta = ( obj_t )BLIS_OBJECT_INITIALIZER;
	bli_obj_init_finish( cast ? BLIS_DOUBLE : BLIS_FLOAT, 1, 1,
	                     cast ? ( void* )&f->ad : ( void* )&f->af, 1, 1, &f->alpha );
	bli_obj_init_finish( cast ? BLIS_DOUBLE : BLIS_FLOAT, 1, 1,
	                     cast ? ( void* )&f->bd : ( void* )&f->bf, 1, 1, &f->beta );
	return 0;
}
static int fail( const char* label, const char* reason )
{
	fprintf( stderr, "%s: %s\n", label, reason );
	return 1;
}
static int check_result
     (
       const char* label, fixture_t* f, double alpha, double beta, int triangle
     )
{
	const matrix_t* a = &f->a; const matrix_t* b = &f->b; const matrix_t* c = &f->c;
	if ( memcmp( a->data, a->saved, a->count*a->elem ) ||
	     memcmp( b->data, b->saved, b->count*b->elem ) )
		return fail( label, "input or input guard changed" );
	for ( size_t p = 0; p < c->count; ++p )
		if ( !c->active[p] && memcmp( c->data + p*c->elem, c->saved + p*c->elem, c->elem ) )
			return fail( label, "output padding or guard changed" );
	const double eps = c->dt == BLIS_DOUBLE ? DBL_EPSILON : FLT_EPSILON;
	for ( dim_t i = 0; i < c->m; ++i )
	for ( dim_t j = 0; j < c->n; ++j )
	{
		if ( ( triangle == 1 && i < j ) || ( triangle == 2 && i > j ) )
		{
			const size_t p = index_at( c, i, j )*c->elem;
			if ( memcmp( c->data + p, c->saved + p, c->elem ) )
				return fail( label, "GEMMT unstored triangle changed" );
			continue;
		}
		value_t sum = { 0, 0 };
		double sumabs = 0;
		if ( alpha != 0.0 )
			for ( dim_t p = 0; p < a->n; ++p )
			{
				const value_t av = load( a, i, p, 1 ), bv = load( b, p, j, 1 );
				sum = add( sum, mul( av, bv ) );
				sumabs += mag( av )*mag( bv );
			}
		const value_t old = beta == 0.0 ? ( value_t ){0, 0} : load( c, i, j, 1 );
		const value_t want = add( mul( ( value_t ){alpha, 0}, sum ),
		                          mul( ( value_t ){beta, 0}, old ) );
		const value_t got = load( c, i, j, 0 );
		// Float inputs are loaded exactly, then every product and accumulation
		// is evaluated in double. The bound allows either fused or unfused
		// accumulation and the extra additions used by complex 1m execution.
		const double factor = c->dt == BLIS_SCOMPLEX ? 16 : 8;
		const double bound = factor*eps*( a->n + 3 )*
		                     ( fabs( alpha )*sumabs + fabs( beta )*mag( old ) + 1 );
		++values;
		if ( !isfinite( got.r ) || !isfinite( got.i ) ||
		     fabs( got.r - want.r ) > bound || fabs( got.i - want.i ) > bound )
		{
			fprintf( stderr, "%s: (%ld,%ld) got=(%.17g,%.17g) expected=(%.17g,%.17g) bound=%.9g\n",
			         label, ( long )i, ( long )j, got.r, got.i, want.r, want.i, bound );
			return 1;
		}
	}
	return 0;
}

static int check_unchanged( const char* label, const fixture_t* f )
{
	const matrix_t* all[] = { &f->a, &f->b, &f->c };
	for ( int x = 0; x < 3; ++x )
		if ( memcmp( all[x]->data, all[x]->saved, all[x]->count*all[x]->elem ) )
			return fail( label, "declined small path wrote matrix storage" );
	return 0;
}

// Check the return code of the small-problem stage. A problem it takes must
// be computed correctly; a problem it declines must be left untouched.
static int check_stage
     (
       const char* label, fixture_t* f, err_t got, int expected,
       double alpha, double beta, int triangle
     )
{
#ifdef BLIS_DISABLE_SUP_HANDLING
	expected = 0;
#endif
	if ( expected )
	{
		if ( got != BLIS_SUCCESS )
			return fail( label, "small path declined an eligible problem" );
		++accepted;
		return check_result( label, f, alpha, beta, triangle );
	}
	if ( got != BLIS_FAILURE )
		return fail( label, "small path took an ineligible problem" );
	++declined;
	return check_unchanged( label, f );
}

static int cblas_case
     (
       dim_t m, dim_t n, dim_t k, int order, int ta, int tb,
       float alpha, float beta, int exact
     )
{
	fixture_t f;
	char label[160];
	snprintf( label, sizeof label, "CBLAS %ldx%ldx%ld order=%d trans=%d%d alpha=%g beta=%g%s",
	          ( long )m, ( long )n, ( long )k, order, ta, tb, alpha, beta,
	          exact ? " exact A/B" : "" );
	if ( fixture_init( &f, m, n, k, order, order, order, ta, tb, 0,
	                   BLIS_FLOAT, BLIS_FLOAT, BLIS_FLOAT, alpha, beta, 0, 0, exact ) )
		return fail( label, "allocation failed" );
	const enum CBLAS_TRANSPOSE trans[] = { CblasNoTrans, CblasTrans, CblasConjTrans };
	const inc_t lda = order ? f.a.cs : f.a.rs, ldb = order ? f.b.cs : f.b.rs;
	// With exact extent, an A or B stored as one column in row-major order
	// or one row in column-major order has leading dimension 1, so both of
	// its strides are 1. The handler computes C (or C^T if C is
	// column-stored) in row-major order, where such an operand is one column
	// if it is not transposed and one row if it is. It reads unit row stride
	// as column storage, which needs a column stride of at least the row
	// count, so it declines a single column of two or more rows, and BLIS
	// computes the problem on its conventional path.
	const int eligible = m > 0 && n > 0 && k > 0 &&
	                     m <= 128 && n <= 128 && k <= 128 && alpha != 0.0f &&
	                     !( ta == 0 && lda == 1 && m*k > 1 ) &&
	                     !( tb == 0 && ldb == 1 && k*n > 1 );
	int bad = 0;
#ifdef TEST_SMALL_STACK
	// Vector shapes go to GEMV in the BLAS layer, not to the small path.
	if ( eligible && m > 1 && n > 1 )
#endif
	{
		cblas_sgemm
		(
		  order ? CblasColMajor : CblasRowMajor, trans[ta], trans[tb],
		  m, n, k, alpha, bli_obj_buffer_at_off( &f.a.obj ), lda,
		  bli_obj_buffer_at_off( &f.b.obj ), ldb,
		  beta, bli_obj_buffer_at_off( &f.c.obj ), order ? f.c.cs : f.c.rs
		);
		++cases;
		bad = check_result( label, &f, alpha, beta, 0 );
	}
	fixture_free( &f );
	if ( bad ) return 1;

	// The same operands as objects, passed straight to the handler with the
	// global context. (For vector shapes the BLAS layer calls GEMV instead,
	// so only this direct call reaches the handler with them.)
	if ( fixture_init( &f, m, n, k, order, order, order, ta, tb, 0,
	                   BLIS_FLOAT, BLIS_FLOAT, BLIS_FLOAT, alpha, beta, 0, 0, exact ) )
		return fail( label, "allocation failed" );
	rntm_t rt = BLIS_RNTM_INITIALIZER;
	const err_t got = bli_sgemmsup_wasm32_simd128
	                  ( &f.alpha, &f.a.obj, &f.b.obj, &f.beta, &f.c.obj,
	                    bli_gks_query_cntx(), &rt );
	bad = check_stage( label, &f, got, eligible, alpha, beta, 0 );
	fixture_free( &f );
	return bad;
}

static void replacement_gemm
     (
       dim_t m, dim_t n, dim_t k, const void* alpha, const void* a,
       const void* b, const void* beta, void* c, inc_t rs, inc_t cs,
       const auxinfo_t* data, const cntx_t* cntx
     )
{
	++custom_gemm_hits;
	( ( gemm_ukr_ft )original_gemm )( m, n, k, alpha, a, b, beta, c, rs, cs, data, cntx );
}
typedef void ( *pack_fn_t )
     (
       conj_t, pack_t, dim_t, dim_t, dim_t, dim_t, dim_t, const void*,
       const void*, inc_t, inc_t, void*, inc_t, const void*, const cntx_t*
     );
static void replacement_pack
     (
       conj_t conj, pack_t schema, dim_t dim, dim_t max, dim_t bcast,
       dim_t n, dim_t nmax, const void* kappa, const void* a, inc_t inca,
       inc_t lda, void* p, inc_t ldp, const void* params, const cntx_t* cntx
     )
{
	++custom_pack_hits;
	( ( pack_fn_t )original_pack )( conj, schema, dim, max, bcast, n, nmax, kappa,
	                               a, inca, lda, p, ldp, params, cntx );
}

enum
{
	NORMAL, SUP_OFF, THRESH_OFF, CUSTOM_GEMM, CUSTOM_PACK,
	ATTACH_A, ATTACH_B, ATTACH_C, COMPUTE_DOUBLE, GEMMT_LOWER, GEMMT_UPPER, ONE_M
};
// Apply the per-operand settings of a mode to a fixture.
static void object_mode( fixture_t* f, int mode )
{
	if ( mode == ATTACH_A || mode == ATTACH_B || mode == ATTACH_C )
	{
		obj_t* o = mode == ATTACH_A ? &f->a.obj : mode == ATTACH_B ? &f->b.obj : &f->c.obj;
		*( float* )bli_obj_internal_scalar_buffer( o ) = -0.5f;
	}
	if ( mode == COMPUTE_DOUBLE ) bli_obj_set_comp_prec( BLIS_DOUBLE_PREC, &f->c.obj );
	if ( mode == GEMMT_LOWER ) bli_obj_set_uplo( BLIS_LOWER, &f->c.obj );
	if ( mode == GEMMT_UPPER ) bli_obj_set_uplo( BLIS_UPPER, &f->c.obj );
}

static int object_case
     (
       const char* name, dim_t m, dim_t n, dim_t k,
       int la, int lb, int lc, int ta, int tb, int tc,
       num_t da, num_t db, num_t dc, float alpha, float beta,
       int cast, int mode, int expected
     )
{
	fixture_t f, g;
	if ( fixture_init( &f, m, n, k, la, lb, lc, ta, tb, tc, da, db, dc, alpha, beta, 1, cast, 0 ) )
		return fail( name, "allocation failed" );
	if ( fixture_init( &g, m, n, k, la, lb, lc, ta, tb, tc, da, db, dc, alpha, beta, 1, cast, 0 ) )
	{
		fixture_free( &f ); return fail( name, "allocation failed" );
	}
	rntm_t runtime = BLIS_RNTM_INITIALIZER;
	double ae = dc == BLIS_DOUBLE && cast ? f.ad : alpha;
	double be = dc == BLIS_DOUBLE && cast ? f.bd : beta;
	if ( mode == ATTACH_C ) be = ( float )( beta * -0.5f );
	if ( mode == ATTACH_A || mode == ATTACH_B ) ae = ( float )( alpha * -0.5f );
	object_mode( &f, mode );
	object_mode( &g, mode );
	const dim_t threshold = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_MT, &context );
	int old_1m = 0, old_nat = 0;
	if ( mode == SUP_OFF ) bli_rntm_disable_l3_sup( &runtime );
	if ( mode == THRESH_OFF ) bli_cntx_set_blksz_def_dt( BLIS_FLOAT, BLIS_MT, 0, &context );
	if ( mode == CUSTOM_GEMM )
		bli_cntx_set_ukr2_dt( ( void_fp )replacement_gemm, BLIS_FLOAT, BLIS_FLOAT, BLIS_GEMM_UKR, &context );
	if ( mode == CUSTOM_PACK )
		bli_cntx_set_ukr2_dt( ( void_fp )replacement_pack, BLIS_FLOAT, BLIS_FLOAT, BLIS_PACKM_KER, &context );
	const int triangle = mode == GEMMT_LOWER ? 1 : mode == GEMMT_UPPER ? 2 : 0;
	if ( mode == ONE_M )
	{
		old_1m = bli_l3_ind_oper_get_enable( BLIS_GEMM, BLIS_1M, BLIS_SCOMPLEX );
		old_nat = bli_l3_ind_oper_get_enable( BLIS_GEMM, BLIS_NAT, BLIS_SCOMPLEX );
		bli_ind_oper_enable_only( BLIS_GEMM, BLIS_1M, BLIS_SCOMPLEX );
		if ( bli_gemmind_find_avail( BLIS_SCOMPLEX ) != BLIS_1M )
		{
			fixture_free( &f ); fixture_free( &g ); return fail( name, "1m is unavailable" );
		}
	}
	const uint64_t gm_before = custom_gemm_hits, pk_before = custom_pack_hits;
	int run_public = 1;
#ifdef TEST_SMALL_STACK
	run_public = expected;
#endif
	if ( run_public )
	{
		if ( triangle ) bli_gemmt_ex( &f.alpha, &f.a.obj, &f.b.obj, &f.beta, &f.c.obj, &context, &runtime );
		else bli_gemm_ex( &f.alpha, &f.a.obj, &f.b.obj, &f.beta, &f.c.obj, &context, &runtime );
		++cases;
	}
	const int gm_used = custom_gemm_hits != gm_before, pk_used = custom_pack_hits != pk_before;
	// The small-problem stage that the public call runs first, on fresh
	// operands and with the same context and runtime.
	const err_t stage = triangle
	    ? bli_gemmtsup( &g.alpha, &g.a.obj, &g.b.obj, &g.beta, &g.c.obj, &context, &runtime )
	    : bli_gemmsup( &g.alpha, &g.a.obj, &g.b.obj, &g.beta, &g.c.obj, &context, &runtime );
	bli_cntx_set_blksz_def_dt( BLIS_FLOAT, BLIS_MT, threshold, &context );
	bli_cntx_set_ukr2_dt( original_gemm, BLIS_FLOAT, BLIS_FLOAT, BLIS_GEMM_UKR, &context );
	bli_cntx_set_ukr2_dt( original_pack, BLIS_FLOAT, BLIS_FLOAT, BLIS_PACKM_KER, &context );
	if ( mode == ONE_M )
	{
		bli_l3_ind_oper_set_enable( BLIS_GEMM, BLIS_1M, BLIS_SCOMPLEX, old_1m );
		bli_l3_ind_oper_set_enable( BLIS_GEMM, BLIS_NAT, BLIS_SCOMPLEX, old_nat );
	}
	int bad = ( run_public && check_result( name, &f, ae, be, triangle ) ) ||
	          check_stage( name, &g, stage, expected, ae, be, triangle );
	if ( run_public && mode == CUSTOM_GEMM && !gm_used ) bad = fail( name, "custom GEMM not used by fallback" );
	if ( run_public && mode == CUSTOM_PACK && !pk_used ) bad = fail( name, "custom packer not used by fallback" );
	fixture_free( &f ); fixture_free( &g );
	return bad;
}

static int direct_reject( const char* name, fixture_t* f, cntx_t* cx, rntm_t* rt )
{
	const err_t result = bli_sgemmsup_wasm32_simd128
	                    ( &f->alpha, &f->a.obj, &f->b.obj, &f->beta, &f->c.obj, cx, rt );
	return check_stage( name, f, result, 0, f->af, f->bf, 0 );
}

static int direct_cases( void )
{
	fixture_t f;
	rntm_t rt;
	int bad = 0;
#define PREPARE() do { \
	if ( fixture_init( &f, 4, 8, 3, 0, 0, 0, 0, 0, 0, BLIS_FLOAT, BLIS_FLOAT, BLIS_FLOAT, 1, 0, 1, 0, 0 ) ) return 1; \
	rt = ( rntm_t )BLIS_RNTM_INITIALIZER; \
} while ( 0 )
#define REJECT(name, change) do { \
	PREPARE(); change; bad = direct_reject( name, &f, &context, &rt ); fixture_free( &f ); if ( bad ) return 1; \
} while ( 0 )
	// The unmodified fixture is taken, so each rejection below comes from
	// its one change.
	PREPARE();
	bad = check_stage( "direct baseline", &f,
	                   bli_sgemmsup_wasm32_simd128( &f.alpha, &f.a.obj, &f.b.obj,
	                                                &f.beta, &f.c.obj, &context, &rt ),
	                   1, f.af, f.bf, 0 );
	fixture_free( &f ); if ( bad ) return 1;
	PREPARE(); bad = direct_reject( "null context", &f, NULL, &rt ); fixture_free( &f ); if ( bad ) return 1;
	PREPARE(); bad = direct_reject( "null runtime", &f, &context, NULL ); fixture_free( &f ); if ( bad ) return 1;
	REJECT( "alpha zero", f.af = 0 );
	REJECT( "zero M", bli_obj_set_length( 0, &f.c.obj ) );
	REJECT( "zero N", bli_obj_set_width( 0, &f.c.obj ) );
	REJECT( "zero K", bli_obj_set_width( 0, &f.a.obj ) );
	REJECT( "M over limit", bli_obj_set_length( 129, &f.c.obj ) );
	REJECT( "N over limit", bli_obj_set_width( 129, &f.c.obj ) );
	REJECT( "K over limit", bli_obj_set_width( 129, &f.a.obj ) );
	REJECT( "SUP off", bli_rntm_disable_l3_sup( &rt ) );
	REJECT( "thread count", bli_rntm_set_num_threads_only( 2, &rt ) );
	REJECT( "thread implementation", bli_rntm_set_thread_impl_only( BLIS_POSIX, &rt ) );
	for ( dim_t l = 0; l < BLIS_NUM_LOOPS; ++l )
		REJECT( "thread way", bli_rntm_set_ways_for_only( ( bszid_t )l, 2, &rt ) );
	for ( int obj = 0; obj < 3; ++obj )
	{
		PREPARE();
		obj_t* o = obj == 0 ? &f.a.obj : obj == 1 ? &f.b.obj : &f.c.obj;
		bli_obj_set_comp_prec( BLIS_DOUBLE_PREC, o );
		bad = direct_reject( "object computation precision", &f, &context, &rt ); fixture_free( &f ); if ( bad ) return 1;
		PREPARE(); o = obj == 0 ? &f.a.obj : obj == 1 ? &f.b.obj : &f.c.obj;
		bli_obj_set_pack_schema( BLIS_PACKED_PANELS, o );
		bad = direct_reject( "prepacked object", &f, &context, &rt ); fixture_free( &f ); if ( bad ) return 1;
		PREPARE(); o = obj == 0 ? &f.a.obj : obj == 1 ? &f.b.obj : &f.c.obj;
		bli_obj_set_struc( BLIS_SYMMETRIC, o );
		bad = direct_reject( "structured object", &f, &context, &rt ); fixture_free( &f ); if ( bad ) return 1;
		PREPARE(); o = obj == 0 ? &f.a.obj : obj == 1 ? &f.b.obj : &f.c.obj;
		bli_obj_set_uplo( BLIS_LOWER, o );
		bad = direct_reject( "nondense object", &f, &context, &rt ); fixture_free( &f ); if ( bad ) return 1;
		PREPARE(); o = obj == 0 ? &f.a.obj : obj == 1 ? &f.b.obj : &f.c.obj;
		bli_obj_set_scalar_dt( BLIS_DOUBLE, o );
		bad = direct_reject( "nonfloat attached scalar", &f, &context, &rt ); fixture_free( &f ); if ( bad ) return 1;
	}
	PREPARE();
	bli_obj_init_finish( BLIS_DOUBLE, 1, 1, &f.ad, 1, 1, &f.alpha );
	bad = direct_reject( "uncast double scalar", &f, &context, &rt ); fixture_free( &f ); if ( bad ) return 1;
	PREPARE();
	bli_obj_init_finish( BLIS_DOUBLE, 1, 1, &f.bd, 1, 1, &f.beta );
	bad = direct_reject( "uncast double beta", &f, &context, &rt ); fixture_free( &f ); if ( bad ) return 1;
	REJECT( "nonscalar alpha", bli_obj_set_width( 2, &f.alpha ) );
	REJECT( "nonscalar beta", bli_obj_set_length( 2, &f.beta ) );
	const kerid_t ids[] = { BLIS_MR, BLIS_NR, BLIS_BBM, BLIS_BBN };
	for ( size_t i = 0; i < sizeof ids / sizeof ids[0]; ++i )
	{
		const dim_t saved = bli_cntx_get_blksz_max_dt( BLIS_FLOAT, ids[i], &context );
		PREPARE(); bli_cntx_set_blksz_max_dt( BLIS_FLOAT, ids[i], saved+1, &context );
		bad = direct_reject( "custom packing dimension", &f, &context, &rt );
		bli_cntx_set_blksz_max_dt( BLIS_FLOAT, ids[i], saved, &context );
		fixture_free( &f ); if ( bad ) return 1;
	}
	for ( int i = 0; i < 2; ++i )
	{
		const dim_t saved = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, ids[i], &context );
		PREPARE(); bli_cntx_set_blksz_def_dt( BLIS_FLOAT, ids[i], saved+1, &context );
		bad = direct_reject( "custom register tile", &f, &context, &rt );
		bli_cntx_set_blksz_def_dt( BLIS_FLOAT, ids[i], saved, &context );
		fixture_free( &f ); if ( bad ) return 1;
	}
	PREPARE();
	bli_cntx_set_ukr_pref_dt( false, BLIS_FLOAT, BLIS_GEMM_UKR_ROW_PREF, &context );
	bad = direct_reject( "custom storage preference", &f, &context, &rt );
	bli_cntx_set_ukr_pref_dt( true, BLIS_FLOAT, BLIS_GEMM_UKR_ROW_PREF, &context );
	fixture_free( &f );
#undef REJECT
#undef PREPARE
	return bad;
}

static int public_cases( void )
{
	const dim_t sizes[] = { 1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64 };
	const size_t ns = sizeof sizes / sizeof sizes[0];
	// Every Cartesian combination is tested, including all small rectangular
	// triples. Cycle independent layout/transpose/scalar settings over them.
	for ( size_t i = 0; i < ns; ++i )
	for ( size_t j = 0; j < ns; ++j )
	for ( size_t p = 0; p < ns; ++p )
	{
		const size_t code = ( i*ns + j )*ns + p;
		if ( cblas_case( sizes[i], sizes[j], sizes[p], code%2, (code/2)%3, (code/6)%3,
		                 code%3 ? 1.0f : -0.75f, code%4 ? -0.5f : -0.0f, 0 ) ) return 1;
	}
	const dim_t shapes[][3] = { {1, 3, 4}, {3, 7, 8}, {4, 8, 16}, {7, 16, 31},
	                           {16, 4, 32}, {31, 63, 7}, {64, 32, 63}, {63, 64, 64}, {65, 65, 65}, {127, 65, 33}, {128, 128, 128} };
	const float alphas[] = { 0.0f, -0.0f, 1.0f, -0.75f };
	const float betas[] = { 0.0f, -0.0f, 1.0f, -0.5f };
	for ( size_t s = 0; s < sizeof shapes / sizeof shapes[0]; ++s )
	for ( int order = 0; order < 2; ++order )
	for ( int ta = 0; ta < 3; ++ta )
	for ( int tb = 0; tb < 3; ++tb )
	for ( size_t a = 0; a < 4; ++a )
	for ( size_t b = 0; b < 4; ++b )
		if ( cblas_case( shapes[s][0], shapes[s][1], shapes[s][2], order, ta, tb, alphas[a], betas[b], 0 ) ) return 1;
	for ( int axis = 0; axis < 3; ++axis )
	for ( int order = 0; order < 2; ++order )
	for ( int ta = 0; ta < 2; ++ta )
	for ( int tb = 0; tb < 2; ++tb )
	{
		dim_t dims[] = { 7, 8, 16 }; dims[axis] = 65;
		if ( cblas_case( dims[0], dims[1], dims[2], order, ta, tb, 1, 0, 0 ) ) return 1;
		dims[axis] = 128;
		if ( cblas_case( dims[0], dims[1], dims[2], order, ta, tb, 1, 0, 0 ) ) return 1;
		dims[axis] = 129;
		if ( cblas_case( dims[0], dims[1], dims[2], order, ta, tb, 1, 0, 0 ) ) return 1;
		dims[axis] = 0;
		if ( cblas_case( dims[0], dims[1], dims[2], order, ta, tb, 1, 0, 0 ) ) return 1;
	}
	// Exact-extent A and B for the handler's edge paths: m and n below, at
	// and above the 4-row and 8-column panel widths (including the direct
	// 4x8 and 8x4 tiles), one size above 64, and k with and without a
	// remainder after the 4-step packing loops.
	const dim_t exact_mn[] = { 1, 3, 4, 5, 7, 8, 13, 67 };
	const dim_t exact_k[] = { 1, 4, 9 };
	const size_t ne = sizeof exact_mn / sizeof exact_mn[0];
	for ( size_t i = 0; i < ne; ++i )
	for ( size_t j = 0; j < ne; ++j )
	for ( size_t p = 0; p < sizeof exact_k / sizeof exact_k[0]; ++p )
	for ( int order = 0; order < 2; ++order )
	for ( int ta = 0; ta < 2; ++ta )
	for ( int tb = 0; tb < 2; ++tb )
		if ( cblas_case( exact_mn[i], exact_mn[j], exact_k[p], order, ta, tb,
		                 ( i + j + p ) % 2 ? 1.0f : -0.75f,
		                 ( i + 2*j + p ) % 3 ? -0.5f : 0.0f, 1 ) ) return 1;
	return 0;
}

static int object_cases( void )
{
	const dim_t vector_shapes[][2] = { {1, 8}, {4, 1}, {1, 1} };
	for ( size_t s = 0; s < sizeof vector_shapes / sizeof vector_shapes[0]; ++s )
		if ( object_case( "object vector-shaped GEMM", vector_shapes[s][0], vector_shapes[s][1], 31,
		                  0, 1, 0, 0, 1, 0, BLIS_FLOAT, BLIS_FLOAT, BLIS_FLOAT, 1, 0, 0, NORMAL, 1 ) ) return 1;
	for ( int la = 0; la < 2; ++la )
	for ( int lb = 0; lb < 2; ++lb )
	for ( int lc = 0; lc < 2; ++lc )
	for ( int ta = 0; ta < 4; ++ta )
	for ( int tb = 0; tb < 4; ++tb )
	for ( int tc = 0; tc < 2; ++tc )
		if ( object_case( "offset mixed storage", 7, 16, 31, la, lb, lc, ta, tb, tc,
		                  BLIS_FLOAT, BLIS_FLOAT, BLIS_FLOAT, -0.75f, 0, 0, NORMAL, 1 ) ) return 1;
	for ( int which = 0; which < 3; ++which )
	for ( int layout = 2; layout < 6; ++layout )
		if ( object_case( "general/negative stride fallback", 7, 8, 16,
		                  which==0?layout:0, which==1?layout:0, which==2?layout:0, 1, 0, 0,
		                  BLIS_FLOAT, BLIS_FLOAT, BLIS_FLOAT, -0.75f, -0.5f, 0, NORMAL, 0 ) ) return 1;
	for ( int mode = SUP_OFF; mode <= COMPUTE_DOUBLE; ++mode )
		if ( object_case( "configured fallback", 8, 16, 31, 0, 1, 0, 0, 1, 0,
		                  BLIS_FLOAT, BLIS_FLOAT, BLIS_FLOAT, -0.75f, -0.5f, 0, mode, 0 ) ) return 1;
	if ( object_case( "public scalar casts", 7, 8, 31, 0, 1, 0, 1, 0, 0,
	                  BLIS_FLOAT, BLIS_FLOAT, BLIS_FLOAT, -0.75f, -0.5f, 1, NORMAL, 1 ) ) return 1;
	if ( object_case( "double GEMM fallback", 7, 8, 31, 0, 1, 0, 1, 0, 0,
	                  BLIS_DOUBLE, BLIS_DOUBLE, BLIS_DOUBLE, -0.75f, -0.5f, 1, NORMAL, 0 ) ) return 1;
	if ( object_case( "mixed precision fallback", 7, 8, 31, 0, 1, 0, 1, 0, 0,
	                  BLIS_DOUBLE, BLIS_FLOAT, BLIS_FLOAT, -0.75f, -0.5f, 0, NORMAL, 0 ) ) return 1;
	for ( int mode = GEMMT_LOWER; mode <= GEMMT_UPPER; ++mode )
	for ( int order = 0; order < 2; ++order )
		if ( object_case( "GEMMT triangle fallback", 7, 7, 31, 0, 1, order, 1, 0, 0,
		                  BLIS_FLOAT, BLIS_FLOAT, BLIS_FLOAT, -0.75f, 0, 0, mode, 0 ) ) return 1;
	for ( int ta = 0; ta < 4; ++ta )
	for ( int tb = 0; tb < 4; ++tb )
		if ( object_case( "complex 1m fallback", 7, 8, 16, 0, 1, 0, ta, tb, 0,
		                  BLIS_SCOMPLEX, BLIS_SCOMPLEX, BLIS_SCOMPLEX, -0.75f, -0.5f, 0, ONE_M, 0 ) ) return 1;
	return 0;
}

int main( void )
{
	bli_init();
	bli_cntx_init_wasm32( &context );
	original_gemm = bli_cntx_get_ukr2_dt( BLIS_FLOAT, BLIS_FLOAT, BLIS_GEMM_UKR, &context );
	original_pack = bli_cntx_get_ukr2_dt( BLIS_FLOAT, BLIS_FLOAT, BLIS_PACKM_KER, &context );
	int bad = 0;
#ifndef BLIS_DISABLE_SUP_HANDLING
	// The object cases reach the handler through these registrations.
	if ( bli_cntx_get_l3_sup_handler( BLIS_GEMM, bli_gks_query_cntx() ) !=
	     ( void_fp )bli_sgemmsup_wasm32_simd128 ||
	     bli_cntx_get_l3_sup_handler( BLIS_GEMM, &context ) !=
	     ( void_fp )bli_sgemmsup_wasm32_simd128 )
		bad = fail( "test-small-sup", "the small single GEMM handler is not registered" );
#endif
	bad = bad || public_cases() || object_cases() || direct_cases();
	bli_cntx_free( &context );
	bli_finalize();
	if ( bad ) return 1;
	printf( "test-small-sup: %llu public calls; the small path took %llu "
	        "problems and declined %llu; %llu output values checked, 0 failures\n",
	        ( unsigned long long )cases, ( unsigned long long )accepted,
	        ( unsigned long long )declined, ( unsigned long long )values );
	return 0;
}
