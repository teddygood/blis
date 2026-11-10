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
#include <wasm_simd128.h>

// Small single-precision GEMM handler for the wasm32 configuration.
//
// The context registers bli_sgemmsup_wasm32_simd128() as the BLIS_GEMM sup
// handler and sets the float MT threshold to 129, so BLIS offers it every
// float GEMM with m <= 128 (or n <= 128 after a storage-preference
// transposition). The handler accepts only problems with m, n and k all in
// [1, 128], nonzero alpha, unit stride in one dimension of every operand,
// single-threaded execution and the default wasm32 kernels and blocksizes.
// It returns BLIS_FAILURE for everything else, and BLIS then runs its
// conventional implementation.
//
// Accepted problems are packed into 4x8 (or 4x4) panels and computed with
// the SIMD tiles below, without the conventional path's per-call setup.

#define BLIS_SGEMMSUP_MADD( acc, a, b ) wasm_f32x4_add( (acc), wasm_f32x4_mul( (a), (b) ) )

// Reference float packing kernel instantiated for this configuration.
void bli_sspackm_wasm32_ref
     (
             conj_t  conja,
             pack_t  schema,
             dim_t   cdim,
             dim_t   cdim_max,
             dim_t   cdim_bcast,
             dim_t   n,
             dim_t   n_max,
       const void*   kappa,
       const void*   a, inc_t inca, inc_t lda,
             void*   p,             inc_t ldp,
       const void*   params,
       const cntx_t* cntx
     );

static inline inc_t bli_sgemmsup_wasm32_abs( inc_t x )
{
	return x < 0 ? -x : x;
}

static inline bool bli_sgemmsup_wasm32_rntm_is_single_thread( const rntm_t* rntm )
{
	if ( !bli_rntm_l3_sup( rntm ) ) return false;

	if ( bli_rntm_thread_impl( rntm ) != BLIS_SINGLE ) return false;

	const dim_t nt = bli_rntm_num_threads( rntm );
	if ( nt > 1 || nt == 0 || nt < -1 ) return false;

	for ( dim_t i = 0; i < BLIS_NUM_LOOPS; ++i )
	{
		const dim_t ways = bli_rntm_ways_for( ( bszid_t )i, rntm );
		if ( ways > 1 || ways == 0 || ways < -1 ) return false;
	}

	return true;
}

static inline bool bli_sgemmsup_wasm32_obj_ok( const obj_t* x )
{
	if ( bli_obj_dt( x ) != BLIS_FLOAT ) return false;
	if ( bli_obj_comp_prec( x ) != bli_obj_prec( x ) ) return false;
	if ( bli_obj_pack_schema( x ) != BLIS_NOT_PACKED ) return false;
	if ( !bli_obj_is_general( x ) ) return false;
	if ( !bli_obj_is_dense( x ) ) return false;
	if ( bli_obj_buffer( x ) == NULL ) return false;
	if ( bli_obj_scalar_dt( x ) != BLIS_FLOAT ) return false;

	const float* s = ( const float* )bli_obj_internal_scalar_buffer( x );
	return s != NULL && *s == 1.0f;
}

static inline bool bli_sgemmsup_wasm32_extract_scalar
     (
       const obj_t* s,
       float*       out
     )
{
	if ( s == NULL ) return false;
	if ( bli_obj_length( s ) != 1 || bli_obj_width( s ) != 1 ) return false;
	if ( bli_obj_buffer( s ) == NULL ) return false;
	if ( bli_obj_dt( s ) == BLIS_CONSTANT )
	{
		const float* p = ( const float* )bli_obj_buffer_for_const( BLIS_FLOAT, s );
		if ( p == NULL ) return false;
		*out = *p;
		return true;
	}
	if ( bli_obj_dt( s ) != BLIS_FLOAT ) return false;
	*out = *( const float* )bli_obj_buffer_at_off( s );
	return true;
}

static inline bool bli_sgemmsup_wasm32_storage_ok
     (
       dim_t m,
       dim_t n,
       inc_t rs,
       inc_t cs
     )
{
	if ( rs <= 0 || cs <= 0 ) return false;
	if ( rs == 1 ) return cs >= m;
	if ( cs == 1 ) return rs >= n;
	return false;
}

static inline void bli_sgemmsup_wasm32_extract_view
     (
       const obj_t*  obj,
       const float** buf_out,
       inc_t*        rs_out,
       inc_t*        cs_out
     )
{
	const inc_t rs0  = bli_obj_row_stride( obj );
	const inc_t cs0  = bli_obj_col_stride( obj );
	const dim_t offm = bli_obj_row_off( obj );
	const dim_t offn = bli_obj_col_off( obj );
	const float* p   = ( const float* )bli_obj_buffer( obj ) + offm * rs0 + offn * cs0;
	if ( bli_obj_has_trans( obj ) )
	{
		*buf_out = p;
		*rs_out  = cs0;
		*cs_out  = rs0;
	}
	else
	{
		*buf_out = p;
		*rs_out  = rs0;
		*cs_out  = cs0;
	}
}

static inline void bli_sgemmsup_wasm32_transpose_4x4
     (
       v128_t  r0,
       v128_t  r1,
       v128_t  r2,
       v128_t  r3,
       v128_t* c0,
       v128_t* c1,
       v128_t* c2,
       v128_t* c3
     )
{
	const v128_t t0 = wasm_i32x4_shuffle( r0, r1, 0, 4, 1, 5 );
	const v128_t t1 = wasm_i32x4_shuffle( r2, r3, 0, 4, 1, 5 );
	const v128_t t2 = wasm_i32x4_shuffle( r0, r1, 2, 6, 3, 7 );
	const v128_t t3 = wasm_i32x4_shuffle( r2, r3, 2, 6, 3, 7 );

	*c0 = wasm_i32x4_shuffle( t0, t1, 0, 1, 4, 5 );
	*c1 = wasm_i32x4_shuffle( t0, t1, 2, 3, 6, 7 );
	*c2 = wasm_i32x4_shuffle( t2, t3, 0, 1, 4, 5 );
	*c3 = wasm_i32x4_shuffle( t2, t3, 2, 3, 6, 7 );
}

static inline void bli_sgemmsup_wasm32_store_row_partial
     (
       float* restrict dst,
       v128_t          val,
       dim_t           nr,
       float           beta,
       v128_t          vbeta
     )
{
	if ( nr == 4 )
	{
		if ( beta != 0.0f )
			val = BLIS_SGEMMSUP_MADD( val, wasm_v128_load( dst ), vbeta );
		wasm_v128_store( dst, val );
		return;
	}

	if ( beta == 0.0f )
	{
		wasm_v128_store32_lane( dst + 0, val, 0 );
		if ( nr >= 2 ) wasm_v128_store32_lane( dst + 1, val, 1 );
		if ( nr >= 3 ) wasm_v128_store32_lane( dst + 2, val, 2 );
	}
	else
	{
		dst[0] = wasm_f32x4_extract_lane( val, 0 ) + beta * dst[0];
		if ( nr >= 2 ) dst[1] = wasm_f32x4_extract_lane( val, 1 ) + beta * dst[1];
		if ( nr >= 3 ) dst[2] = wasm_f32x4_extract_lane( val, 2 ) + beta * dst[2];
	}
}

static inline void bli_sgemmsup_wasm32_store_row_4x8
     (
       float* restrict dst,
       v128_t          lo,
       v128_t          hi,
       dim_t           nr,
       float           beta,
       v128_t          vbeta
     )
{
	if ( nr == 8 )
	{
		if ( beta != 0.0f )
		{
			lo = BLIS_SGEMMSUP_MADD( lo, wasm_v128_load( dst + 0 ), vbeta );
			hi = BLIS_SGEMMSUP_MADD( hi, wasm_v128_load( dst + 4 ), vbeta );
		}
		wasm_v128_store( dst + 0, lo );
		wasm_v128_store( dst + 4, hi );
		return;
	}

	if ( beta != 0.0f )
		lo = BLIS_SGEMMSUP_MADD( lo, wasm_v128_load( dst + 0 ), vbeta );
	wasm_v128_store( dst + 0, lo );
	bli_sgemmsup_wasm32_store_row_partial( dst + 4, hi, nr - 4, beta, vbeta );
}

// Packed 4x8 microkernel with compile-time constant strides (stride_a = 4, stride_b = 8)
// and a single K loop using scalar-load splats for A.
static inline void bli_sgemmsup_wasm32_ukernel_4x8_packed
     (
       dim_t                mr,
       dim_t                nr,
       dim_t                k,
       float                alpha,
       const float*         a,
       const float*         b,
       float                beta,
       float* restrict      c,
       inc_t                rs_c
     )
{
	v128_t c00 = wasm_f32x4_const_splat( 0.0f );
	v128_t c01 = wasm_f32x4_const_splat( 0.0f );
	v128_t c10 = wasm_f32x4_const_splat( 0.0f );
	v128_t c11 = wasm_f32x4_const_splat( 0.0f );
	v128_t c20 = wasm_f32x4_const_splat( 0.0f );
	v128_t c21 = wasm_f32x4_const_splat( 0.0f );
	v128_t c30 = wasm_f32x4_const_splat( 0.0f );
	v128_t c31 = wasm_f32x4_const_splat( 0.0f );

	const float* a_p = a;
	const float* b_p = b;

	for ( dim_t p = 0; p < k; ++p )
	{
		const v128_t b0 = wasm_v128_load( b_p );
		const v128_t b1 = wasm_v128_load( b_p + 4 );
		const v128_t a0 = wasm_f32x4_splat( a_p[0] );
		const v128_t a1 = wasm_f32x4_splat( a_p[1] );
		const v128_t a2 = wasm_f32x4_splat( a_p[2] );
		const v128_t a3 = wasm_f32x4_splat( a_p[3] );

		c00 = BLIS_SGEMMSUP_MADD( c00, a0, b0 );
		c01 = BLIS_SGEMMSUP_MADD( c01, a0, b1 );
		c10 = BLIS_SGEMMSUP_MADD( c10, a1, b0 );
		c11 = BLIS_SGEMMSUP_MADD( c11, a1, b1 );
		c20 = BLIS_SGEMMSUP_MADD( c20, a2, b0 );
		c21 = BLIS_SGEMMSUP_MADD( c21, a2, b1 );
		c30 = BLIS_SGEMMSUP_MADD( c30, a3, b0 );
		c31 = BLIS_SGEMMSUP_MADD( c31, a3, b1 );

		a_p += 4;
		b_p += 8;
	}

	if ( alpha != 1.0f )
	{
		const v128_t valpha = wasm_f32x4_splat( alpha );
		c00 = wasm_f32x4_mul( c00, valpha );
		c01 = wasm_f32x4_mul( c01, valpha );
		c10 = wasm_f32x4_mul( c10, valpha );
		c11 = wasm_f32x4_mul( c11, valpha );
		c20 = wasm_f32x4_mul( c20, valpha );
		c21 = wasm_f32x4_mul( c21, valpha );
		c30 = wasm_f32x4_mul( c30, valpha );
		c31 = wasm_f32x4_mul( c31, valpha );
	}

	const v128_t vbeta = wasm_f32x4_splat( beta );
	if ( mr == 4 && nr == 8 )
	{
		if ( beta != 0.0f )
		{
			c00 = BLIS_SGEMMSUP_MADD( c00, wasm_v128_load( c + 0 * rs_c + 0 ), vbeta );
			c01 = BLIS_SGEMMSUP_MADD( c01, wasm_v128_load( c + 0 * rs_c + 4 ), vbeta );
			c10 = BLIS_SGEMMSUP_MADD( c10, wasm_v128_load( c + 1 * rs_c + 0 ), vbeta );
			c11 = BLIS_SGEMMSUP_MADD( c11, wasm_v128_load( c + 1 * rs_c + 4 ), vbeta );
			c20 = BLIS_SGEMMSUP_MADD( c20, wasm_v128_load( c + 2 * rs_c + 0 ), vbeta );
			c21 = BLIS_SGEMMSUP_MADD( c21, wasm_v128_load( c + 2 * rs_c + 4 ), vbeta );
			c30 = BLIS_SGEMMSUP_MADD( c30, wasm_v128_load( c + 3 * rs_c + 0 ), vbeta );
			c31 = BLIS_SGEMMSUP_MADD( c31, wasm_v128_load( c + 3 * rs_c + 4 ), vbeta );
		}
		wasm_v128_store( c + 0 * rs_c + 0, c00 );
		wasm_v128_store( c + 0 * rs_c + 4, c01 );
		wasm_v128_store( c + 1 * rs_c + 0, c10 );
		wasm_v128_store( c + 1 * rs_c + 4, c11 );
		wasm_v128_store( c + 2 * rs_c + 0, c20 );
		wasm_v128_store( c + 2 * rs_c + 4, c21 );
		wasm_v128_store( c + 3 * rs_c + 0, c30 );
		wasm_v128_store( c + 3 * rs_c + 4, c31 );
		return;
	}

	bli_sgemmsup_wasm32_store_row_4x8( c + 0 * rs_c, c00, c01, nr, beta, vbeta );
	if ( mr >= 2 ) bli_sgemmsup_wasm32_store_row_4x8( c + 1 * rs_c, c10, c11, nr, beta, vbeta );
	if ( mr >= 3 ) bli_sgemmsup_wasm32_store_row_4x8( c + 2 * rs_c, c20, c21, nr, beta, vbeta );
	if ( mr >= 4 ) bli_sgemmsup_wasm32_store_row_4x8( c + 3 * rs_c, c30, c31, nr, beta, vbeta );
}

// Packed 4x4 microkernel for nr <= 4 with compile-time constant strides (stride_a = 4, stride_b = 4)
// and a single K loop using scalar-load splats for A.
static inline void bli_sgemmsup_wasm32_ukernel_4x4_packed
     (
       dim_t                mr,
       dim_t                nr,
       dim_t                k,
       float                alpha,
       const float*         a,
       const float*         b,
       float                beta,
       float* restrict      c,
       inc_t                rs_c
     )
{
	v128_t c00 = wasm_f32x4_const_splat( 0.0f );
	v128_t c10 = wasm_f32x4_const_splat( 0.0f );
	v128_t c20 = wasm_f32x4_const_splat( 0.0f );
	v128_t c30 = wasm_f32x4_const_splat( 0.0f );

	const float* a_p = a;
	const float* b_p = b;

	for ( dim_t p = 0; p < k; ++p )
	{
		const v128_t b0 = wasm_v128_load( b_p );
		const v128_t a0 = wasm_f32x4_splat( a_p[0] );
		const v128_t a1 = wasm_f32x4_splat( a_p[1] );
		const v128_t a2 = wasm_f32x4_splat( a_p[2] );
		const v128_t a3 = wasm_f32x4_splat( a_p[3] );

		c00 = BLIS_SGEMMSUP_MADD( c00, a0, b0 );
		c10 = BLIS_SGEMMSUP_MADD( c10, a1, b0 );
		c20 = BLIS_SGEMMSUP_MADD( c20, a2, b0 );
		c30 = BLIS_SGEMMSUP_MADD( c30, a3, b0 );

		a_p += 4;
		b_p += 4;
	}

	if ( alpha != 1.0f )
	{
		const v128_t valpha = wasm_f32x4_splat( alpha );
		c00 = wasm_f32x4_mul( c00, valpha );
		c10 = wasm_f32x4_mul( c10, valpha );
		c20 = wasm_f32x4_mul( c20, valpha );
		c30 = wasm_f32x4_mul( c30, valpha );
	}

	const v128_t vbeta = wasm_f32x4_splat( beta );
	bli_sgemmsup_wasm32_store_row_partial( c + 0 * rs_c, c00, nr, beta, vbeta );
	if ( mr >= 2 ) bli_sgemmsup_wasm32_store_row_partial( c + 1 * rs_c, c10, nr, beta, vbeta );
	if ( mr >= 3 ) bli_sgemmsup_wasm32_store_row_partial( c + 2 * rs_c, c20, nr, beta, vbeta );
	if ( mr >= 4 ) bli_sgemmsup_wasm32_store_row_partial( c + 3 * rs_c, c30, nr, beta, vbeta );
}

// Direct unpacked 4x8 microkernel for single-panel shapes (m <= 4, n <= 8).
static inline void bli_sgemmsup_wasm32_direct_4x8
     (
       dim_t                mr,
       dim_t                nr,
       dim_t                k,
       float                alpha,
       const float*         a,
       inc_t                rs_a,
       inc_t                cs_a,
       const float*         b,
       inc_t                rs_b,
       inc_t                cs_b,
       float                beta,
       float* restrict      c,
       inc_t                rs_c
     )
{
	v128_t c00 = wasm_f32x4_const_splat( 0.0f );
	v128_t c01 = wasm_f32x4_const_splat( 0.0f );
	v128_t c10 = wasm_f32x4_const_splat( 0.0f );
	v128_t c11 = wasm_f32x4_const_splat( 0.0f );
	v128_t c20 = wasm_f32x4_const_splat( 0.0f );
	v128_t c21 = wasm_f32x4_const_splat( 0.0f );
	v128_t c30 = wasm_f32x4_const_splat( 0.0f );
	v128_t c31 = wasm_f32x4_const_splat( 0.0f );

	if ( cs_b == 1 && nr == 8 )
	{
		if ( cs_a == 1 )
		{
			const float* a0 = a;
			const float* a1 = mr >= 2 ? a + rs_a : a0;
			const float* a2 = mr >= 3 ? a + 2 * rs_a : a0;
			const float* a3 = mr >= 4 ? a + 3 * rs_a : a0;
			const float* b_p = b;
			for ( dim_t p = 0; p < k; ++p, b_p += rs_b )
			{
				const v128_t b0 = wasm_v128_load( b_p + 0 );
				const v128_t b1 = wasm_v128_load( b_p + 4 );
				const v128_t v0 = wasm_f32x4_splat( a0[p] );
				const v128_t v1 = wasm_f32x4_splat( a1[p] );
				const v128_t v2 = wasm_f32x4_splat( a2[p] );
				const v128_t v3 = wasm_f32x4_splat( a3[p] );
				c00 = BLIS_SGEMMSUP_MADD( c00, v0, b0 );
				c01 = BLIS_SGEMMSUP_MADD( c01, v0, b1 );
				c10 = BLIS_SGEMMSUP_MADD( c10, v1, b0 );
				c11 = BLIS_SGEMMSUP_MADD( c11, v1, b1 );
				c20 = BLIS_SGEMMSUP_MADD( c20, v2, b0 );
				c21 = BLIS_SGEMMSUP_MADD( c21, v2, b1 );
				c30 = BLIS_SGEMMSUP_MADD( c30, v3, b0 );
				c31 = BLIS_SGEMMSUP_MADD( c31, v3, b1 );
			}
		}
		else // rs_a == 1
		{
			const float* a_p = a;
			const float* b_p = b;
			for ( dim_t p = 0; p < k; ++p, a_p += cs_a, b_p += rs_b )
			{
				const v128_t b0 = wasm_v128_load( b_p + 0 );
				const v128_t b1 = wasm_v128_load( b_p + 4 );
				const v128_t v0 = wasm_f32x4_splat( a_p[0] );
				const v128_t v1 = mr >= 2 ? wasm_f32x4_splat( a_p[1] ) : v0;
				const v128_t v2 = mr >= 3 ? wasm_f32x4_splat( a_p[2] ) : v0;
				const v128_t v3 = mr >= 4 ? wasm_f32x4_splat( a_p[3] ) : v0;
				c00 = BLIS_SGEMMSUP_MADD( c00, v0, b0 );
				c01 = BLIS_SGEMMSUP_MADD( c01, v0, b1 );
				c10 = BLIS_SGEMMSUP_MADD( c10, v1, b0 );
				c11 = BLIS_SGEMMSUP_MADD( c11, v1, b1 );
				c20 = BLIS_SGEMMSUP_MADD( c20, v2, b0 );
				c21 = BLIS_SGEMMSUP_MADD( c21, v2, b1 );
				c30 = BLIS_SGEMMSUP_MADD( c30, v3, b0 );
				c31 = BLIS_SGEMMSUP_MADD( c31, v3, b1 );
			}
		}
	}
	else // rs_b == 1 && nr == 8
	{
		const float* b0_col = b + 0 * cs_b;
		const float* b1_col = b + 1 * cs_b;
		const float* b2_col = b + 2 * cs_b;
		const float* b3_col = b + 3 * cs_b;
		const float* b4_col = b + 4 * cs_b;
		const float* b5_col = b + 5 * cs_b;
		const float* b6_col = b + 6 * cs_b;
		const float* b7_col = b + 7 * cs_b;
		dim_t p = 0;
		for ( ; p + 4 <= k; p += 4 )
		{
			v128_t b00, b10, b20, b30, b01, b11, b21, b31;
			bli_sgemmsup_wasm32_transpose_4x4
			(
			  wasm_v128_load( b0_col + p ),
			  wasm_v128_load( b1_col + p ),
			  wasm_v128_load( b2_col + p ),
			  wasm_v128_load( b3_col + p ),
			  &b00, &b10, &b20, &b30
			);
			bli_sgemmsup_wasm32_transpose_4x4
			(
			  wasm_v128_load( b4_col + p ),
			  wasm_v128_load( b5_col + p ),
			  wasm_v128_load( b6_col + p ),
			  wasm_v128_load( b7_col + p ),
			  &b01, &b11, &b21, &b31
			);
			for ( dim_t s = 0; s < 4; ++s )
			{
				const v128_t bl = s == 0 ? b00 : ( s == 1 ? b10 : ( s == 2 ? b20 : b30 ) );
				const v128_t bh = s == 0 ? b01 : ( s == 1 ? b11 : ( s == 2 ? b21 : b31 ) );
				const float* ap = a + ( p + s ) * cs_a;
				const v128_t v0 = wasm_f32x4_splat( ap[0] );
				const v128_t v1 = mr >= 2 ? wasm_f32x4_splat( ap[1 * rs_a] ) : v0;
				const v128_t v2 = mr >= 3 ? wasm_f32x4_splat( ap[2 * rs_a] ) : v0;
				const v128_t v3 = mr >= 4 ? wasm_f32x4_splat( ap[3 * rs_a] ) : v0;
				c00 = BLIS_SGEMMSUP_MADD( c00, v0, bl );
				c01 = BLIS_SGEMMSUP_MADD( c01, v0, bh );
				c10 = BLIS_SGEMMSUP_MADD( c10, v1, bl );
				c11 = BLIS_SGEMMSUP_MADD( c11, v1, bh );
				c20 = BLIS_SGEMMSUP_MADD( c20, v2, bl );
				c21 = BLIS_SGEMMSUP_MADD( c21, v2, bh );
				c30 = BLIS_SGEMMSUP_MADD( c30, v3, bl );
				c31 = BLIS_SGEMMSUP_MADD( c31, v3, bh );
			}
		}
		for ( ; p < k; ++p )
		{
			const v128_t bl = wasm_f32x4_make( b0_col[p], b1_col[p], b2_col[p], b3_col[p] );
			const v128_t bh = wasm_f32x4_make( b4_col[p], b5_col[p], b6_col[p], b7_col[p] );
			const float* ap = a + p * cs_a;
			const v128_t v0 = wasm_f32x4_splat( ap[0] );
			const v128_t v1 = mr >= 2 ? wasm_f32x4_splat( ap[1 * rs_a] ) : v0;
			const v128_t v2 = mr >= 3 ? wasm_f32x4_splat( ap[2 * rs_a] ) : v0;
			const v128_t v3 = mr >= 4 ? wasm_f32x4_splat( ap[3 * rs_a] ) : v0;
			c00 = BLIS_SGEMMSUP_MADD( c00, v0, bl );
			c01 = BLIS_SGEMMSUP_MADD( c01, v0, bh );
			c10 = BLIS_SGEMMSUP_MADD( c10, v1, bl );
			c11 = BLIS_SGEMMSUP_MADD( c11, v1, bh );
			c20 = BLIS_SGEMMSUP_MADD( c20, v2, bl );
			c21 = BLIS_SGEMMSUP_MADD( c21, v2, bh );
			c30 = BLIS_SGEMMSUP_MADD( c30, v3, bl );
			c31 = BLIS_SGEMMSUP_MADD( c31, v3, bh );
		}
	}

	if ( alpha != 1.0f )
	{
		const v128_t valpha = wasm_f32x4_splat( alpha );
		c00 = wasm_f32x4_mul( c00, valpha );
		c01 = wasm_f32x4_mul( c01, valpha );
		c10 = wasm_f32x4_mul( c10, valpha );
		c11 = wasm_f32x4_mul( c11, valpha );
		c20 = wasm_f32x4_mul( c20, valpha );
		c21 = wasm_f32x4_mul( c21, valpha );
		c30 = wasm_f32x4_mul( c30, valpha );
		c31 = wasm_f32x4_mul( c31, valpha );
	}

	const v128_t vbeta = wasm_f32x4_splat( beta );
	bli_sgemmsup_wasm32_store_row_4x8( c + 0 * rs_c, c00, c01, nr, beta, vbeta );
	if ( mr >= 2 ) bli_sgemmsup_wasm32_store_row_4x8( c + 1 * rs_c, c10, c11, nr, beta, vbeta );
	if ( mr >= 3 ) bli_sgemmsup_wasm32_store_row_4x8( c + 2 * rs_c, c20, c21, nr, beta, vbeta );
	if ( mr >= 4 ) bli_sgemmsup_wasm32_store_row_4x8( c + 3 * rs_c, c30, c31, nr, beta, vbeta );
}

// Direct unpacked 8x4 microkernel for m == 8, n == 4 shapes (8x17x4-C and transposed 4x17x8-F).
static inline void bli_sgemmsup_wasm32_direct_8x4
     (
       dim_t                k,
       float                alpha,
       const float*         a,
       inc_t                rs_a,
       inc_t                cs_a,
       const float*         b,
       inc_t                rs_b,
       inc_t                cs_b,
       float                beta,
       float* restrict      c,
       inc_t                rs_c
     )
{
	v128_t c0 = wasm_f32x4_const_splat( 0.0f );
	v128_t c1 = wasm_f32x4_const_splat( 0.0f );
	v128_t c2 = wasm_f32x4_const_splat( 0.0f );
	v128_t c3 = wasm_f32x4_const_splat( 0.0f );
	v128_t c4 = wasm_f32x4_const_splat( 0.0f );
	v128_t c5 = wasm_f32x4_const_splat( 0.0f );
	v128_t c6 = wasm_f32x4_const_splat( 0.0f );
	v128_t c7 = wasm_f32x4_const_splat( 0.0f );

	if ( cs_b == 1 )
	{
		const float* b_p = b;
		for ( dim_t p = 0; p < k; ++p, b_p += rs_b )
		{
			const v128_t bv = wasm_v128_load( b_p );
			const float* ap = a + p * cs_a;
			c0 = BLIS_SGEMMSUP_MADD( c0, wasm_f32x4_splat( ap[0 * rs_a] ), bv );
			c1 = BLIS_SGEMMSUP_MADD( c1, wasm_f32x4_splat( ap[1 * rs_a] ), bv );
			c2 = BLIS_SGEMMSUP_MADD( c2, wasm_f32x4_splat( ap[2 * rs_a] ), bv );
			c3 = BLIS_SGEMMSUP_MADD( c3, wasm_f32x4_splat( ap[3 * rs_a] ), bv );
			c4 = BLIS_SGEMMSUP_MADD( c4, wasm_f32x4_splat( ap[4 * rs_a] ), bv );
			c5 = BLIS_SGEMMSUP_MADD( c5, wasm_f32x4_splat( ap[5 * rs_a] ), bv );
			c6 = BLIS_SGEMMSUP_MADD( c6, wasm_f32x4_splat( ap[6 * rs_a] ), bv );
			c7 = BLIS_SGEMMSUP_MADD( c7, wasm_f32x4_splat( ap[7 * rs_a] ), bv );
		}
	}
	else // rs_b == 1
	{
		const float* b0_col = b + 0 * cs_b;
		const float* b1_col = b + 1 * cs_b;
		const float* b2_col = b + 2 * cs_b;
		const float* b3_col = b + 3 * cs_b;
		dim_t p = 0;
		for ( ; p + 4 <= k; p += 4 )
		{
			v128_t b0, b1, b2, b3;
			bli_sgemmsup_wasm32_transpose_4x4
			(
			  wasm_v128_load( b0_col + p ),
			  wasm_v128_load( b1_col + p ),
			  wasm_v128_load( b2_col + p ),
			  wasm_v128_load( b3_col + p ),
			  &b0, &b1, &b2, &b3
			);
			for ( dim_t s = 0; s < 4; ++s )
			{
				const v128_t bv = s == 0 ? b0 : ( s == 1 ? b1 : ( s == 2 ? b2 : b3 ) );
				const float* ap = a + ( p + s ) * cs_a;
				c0 = BLIS_SGEMMSUP_MADD( c0, wasm_f32x4_splat( ap[0 * rs_a] ), bv );
				c1 = BLIS_SGEMMSUP_MADD( c1, wasm_f32x4_splat( ap[1 * rs_a] ), bv );
				c2 = BLIS_SGEMMSUP_MADD( c2, wasm_f32x4_splat( ap[2 * rs_a] ), bv );
				c3 = BLIS_SGEMMSUP_MADD( c3, wasm_f32x4_splat( ap[3 * rs_a] ), bv );
				c4 = BLIS_SGEMMSUP_MADD( c4, wasm_f32x4_splat( ap[4 * rs_a] ), bv );
				c5 = BLIS_SGEMMSUP_MADD( c5, wasm_f32x4_splat( ap[5 * rs_a] ), bv );
				c6 = BLIS_SGEMMSUP_MADD( c6, wasm_f32x4_splat( ap[6 * rs_a] ), bv );
				c7 = BLIS_SGEMMSUP_MADD( c7, wasm_f32x4_splat( ap[7 * rs_a] ), bv );
			}
		}
		for ( ; p < k; ++p )
		{
			const v128_t bv = wasm_f32x4_make( b0_col[p], b1_col[p], b2_col[p], b3_col[p] );
			const float* ap = a + p * cs_a;
			c0 = BLIS_SGEMMSUP_MADD( c0, wasm_f32x4_splat( ap[0 * rs_a] ), bv );
			c1 = BLIS_SGEMMSUP_MADD( c1, wasm_f32x4_splat( ap[1 * rs_a] ), bv );
			c2 = BLIS_SGEMMSUP_MADD( c2, wasm_f32x4_splat( ap[2 * rs_a] ), bv );
			c3 = BLIS_SGEMMSUP_MADD( c3, wasm_f32x4_splat( ap[3 * rs_a] ), bv );
			c4 = BLIS_SGEMMSUP_MADD( c4, wasm_f32x4_splat( ap[4 * rs_a] ), bv );
			c5 = BLIS_SGEMMSUP_MADD( c5, wasm_f32x4_splat( ap[5 * rs_a] ), bv );
			c6 = BLIS_SGEMMSUP_MADD( c6, wasm_f32x4_splat( ap[6 * rs_a] ), bv );
			c7 = BLIS_SGEMMSUP_MADD( c7, wasm_f32x4_splat( ap[7 * rs_a] ), bv );
		}
	}

	if ( alpha != 1.0f )
	{
		const v128_t valpha = wasm_f32x4_splat( alpha );
		c0 = wasm_f32x4_mul( c0, valpha );
		c1 = wasm_f32x4_mul( c1, valpha );
		c2 = wasm_f32x4_mul( c2, valpha );
		c3 = wasm_f32x4_mul( c3, valpha );
		c4 = wasm_f32x4_mul( c4, valpha );
		c5 = wasm_f32x4_mul( c5, valpha );
		c6 = wasm_f32x4_mul( c6, valpha );
		c7 = wasm_f32x4_mul( c7, valpha );
	}

	const v128_t vbeta = wasm_f32x4_splat( beta );
	if ( beta != 0.0f )
	{
		c0 = BLIS_SGEMMSUP_MADD( c0, wasm_v128_load( c + 0 * rs_c ), vbeta );
		c1 = BLIS_SGEMMSUP_MADD( c1, wasm_v128_load( c + 1 * rs_c ), vbeta );
		c2 = BLIS_SGEMMSUP_MADD( c2, wasm_v128_load( c + 2 * rs_c ), vbeta );
		c3 = BLIS_SGEMMSUP_MADD( c3, wasm_v128_load( c + 3 * rs_c ), vbeta );
		c4 = BLIS_SGEMMSUP_MADD( c4, wasm_v128_load( c + 4 * rs_c ), vbeta );
		c5 = BLIS_SGEMMSUP_MADD( c5, wasm_v128_load( c + 5 * rs_c ), vbeta );
		c6 = BLIS_SGEMMSUP_MADD( c6, wasm_v128_load( c + 6 * rs_c ), vbeta );
		c7 = BLIS_SGEMMSUP_MADD( c7, wasm_v128_load( c + 7 * rs_c ), vbeta );
	}
	wasm_v128_store( c + 0 * rs_c, c0 );
	wasm_v128_store( c + 1 * rs_c, c1 );
	wasm_v128_store( c + 2 * rs_c, c2 );
	wasm_v128_store( c + 3 * rs_c, c3 );
	wasm_v128_store( c + 4 * rs_c, c4 );
	wasm_v128_store( c + 5 * rs_c, c5 );
	wasm_v128_store( c + 6 * rs_c, c6 );
	wasm_v128_store( c + 7 * rs_c, c7 );
}

static inline void bli_sgemmsup_wasm32_pack_a_panel
     (
       dim_t           mr,
       dim_t           k,
       const float*    a,
       inc_t           rs_a,
       inc_t           cs_a,
       float* restrict dst
     )
{
	if ( mr == 4 )
	{
		if ( rs_a == 1 )
		{
			dim_t p = 0;
			for ( ; p + 4 <= k; p += 4 )
			{
				wasm_v128_store( dst + ( p + 0 ) * 4, wasm_v128_load( a + ( p + 0 ) * cs_a ) );
				wasm_v128_store( dst + ( p + 1 ) * 4, wasm_v128_load( a + ( p + 1 ) * cs_a ) );
				wasm_v128_store( dst + ( p + 2 ) * 4, wasm_v128_load( a + ( p + 2 ) * cs_a ) );
				wasm_v128_store( dst + ( p + 3 ) * 4, wasm_v128_load( a + ( p + 3 ) * cs_a ) );
			}
			for ( ; p < k; ++p )
				wasm_v128_store( dst + p * 4, wasm_v128_load( a + p * cs_a ) );
			return;
		}
		if ( cs_a == 1 )
		{
			const float* a0 = a + 0 * rs_a;
			const float* a1 = a + 1 * rs_a;
			const float* a2 = a + 2 * rs_a;
			const float* a3 = a + 3 * rs_a;
			dim_t p = 0;
			for ( ; p + 4 <= k; p += 4 )
			{
				v128_t c0, c1, c2, c3;
				bli_sgemmsup_wasm32_transpose_4x4
				(
				  wasm_v128_load( a0 + p ),
				  wasm_v128_load( a1 + p ),
				  wasm_v128_load( a2 + p ),
				  wasm_v128_load( a3 + p ),
				  &c0, &c1, &c2, &c3
				);
				wasm_v128_store( dst + ( p + 0 ) * 4, c0 );
				wasm_v128_store( dst + ( p + 1 ) * 4, c1 );
				wasm_v128_store( dst + ( p + 2 ) * 4, c2 );
				wasm_v128_store( dst + ( p + 3 ) * 4, c3 );
			}
			for ( ; p < k; ++p )
			{
				dst[p * 4 + 0] = a0[p];
				dst[p * 4 + 1] = a1[p];
				dst[p * 4 + 2] = a2[p];
				dst[p * 4 + 3] = a3[p];
			}
			return;
		}
	}

	if ( cs_a == 1 )
	{
		const v128_t vzero = wasm_f32x4_const_splat( 0.0f );
		const float* a0 = a;
		const float* a1 = mr >= 2 ? a + 1 * rs_a : NULL;
		const float* a2 = mr >= 3 ? a + 2 * rs_a : NULL;
		dim_t p = 0;
		for ( ; p + 4 <= k; p += 4 )
		{
			v128_t c0, c1, c2, c3;
			bli_sgemmsup_wasm32_transpose_4x4
			(
			  wasm_v128_load( a0 + p ),
			  a1 ? wasm_v128_load( a1 + p ) : vzero,
			  a2 ? wasm_v128_load( a2 + p ) : vzero,
			  vzero,
			  &c0, &c1, &c2, &c3
			);
			wasm_v128_store( dst + ( p + 0 ) * 4, c0 );
			wasm_v128_store( dst + ( p + 1 ) * 4, c1 );
			wasm_v128_store( dst + ( p + 2 ) * 4, c2 );
			wasm_v128_store( dst + ( p + 3 ) * 4, c3 );
		}
		for ( ; p < k; ++p )
		{
			dst[p * 4 + 0] = a0[p];
			dst[p * 4 + 1] = a1 ? a1[p] : 0.0f;
			dst[p * 4 + 2] = a2 ? a2[p] : 0.0f;
			dst[p * 4 + 3] = 0.0f;
		}
		return;
	}

	for ( dim_t p = 0; p < k; ++p )
	{
		const float* src = a + p * cs_a;
		float*       out = dst + p * 4;
		out[0] = src[0];
		out[1] = mr >= 2 ? src[1 * rs_a] : 0.0f;
		out[2] = mr >= 3 ? src[2 * rs_a] : 0.0f;
		out[3] = 0.0f;
	}
}

static inline void bli_sgemmsup_wasm32_pack_b_panel
     (
       dim_t           nr,
       dim_t           k,
       const float*    b,
       inc_t           rs_b,
       inc_t           cs_b,
       float* restrict dst
     )
{
	if ( nr == 8 )
	{
		if ( cs_b == 1 )
		{
			dim_t p = 0;
			for ( ; p + 4 <= k; p += 4 )
			{
				const float* s0 = b + ( p + 0 ) * rs_b;
				const float* s1 = b + ( p + 1 ) * rs_b;
				const float* s2 = b + ( p + 2 ) * rs_b;
				const float* s3 = b + ( p + 3 ) * rs_b;
				wasm_v128_store( dst + ( p + 0 ) * 8 + 0, wasm_v128_load( s0 + 0 ) );
				wasm_v128_store( dst + ( p + 0 ) * 8 + 4, wasm_v128_load( s0 + 4 ) );
				wasm_v128_store( dst + ( p + 1 ) * 8 + 0, wasm_v128_load( s1 + 0 ) );
				wasm_v128_store( dst + ( p + 1 ) * 8 + 4, wasm_v128_load( s1 + 4 ) );
				wasm_v128_store( dst + ( p + 2 ) * 8 + 0, wasm_v128_load( s2 + 0 ) );
				wasm_v128_store( dst + ( p + 2 ) * 8 + 4, wasm_v128_load( s2 + 4 ) );
				wasm_v128_store( dst + ( p + 3 ) * 8 + 0, wasm_v128_load( s3 + 0 ) );
				wasm_v128_store( dst + ( p + 3 ) * 8 + 4, wasm_v128_load( s3 + 4 ) );
			}
			for ( ; p < k; ++p )
			{
				const float* src = b + p * rs_b;
				wasm_v128_store( dst + p * 8 + 0, wasm_v128_load( src + 0 ) );
				wasm_v128_store( dst + p * 8 + 4, wasm_v128_load( src + 4 ) );
			}
			return;
		}
		if ( rs_b == 1 )
		{
			const float* b0 = b + 0 * cs_b;
			const float* b1 = b + 1 * cs_b;
			const float* b2 = b + 2 * cs_b;
			const float* b3 = b + 3 * cs_b;
			const float* b4 = b + 4 * cs_b;
			const float* b5 = b + 5 * cs_b;
			const float* b6 = b + 6 * cs_b;
			const float* b7 = b + 7 * cs_b;
			dim_t p = 0;
			for ( ; p + 4 <= k; p += 4 )
			{
				v128_t c0, c1, c2, c3, d0, d1, d2, d3;
				bli_sgemmsup_wasm32_transpose_4x4
				(
				  wasm_v128_load( b0 + p ),
				  wasm_v128_load( b1 + p ),
				  wasm_v128_load( b2 + p ),
				  wasm_v128_load( b3 + p ),
				  &c0, &c1, &c2, &c3
				);
				bli_sgemmsup_wasm32_transpose_4x4
				(
				  wasm_v128_load( b4 + p ),
				  wasm_v128_load( b5 + p ),
				  wasm_v128_load( b6 + p ),
				  wasm_v128_load( b7 + p ),
				  &d0, &d1, &d2, &d3
				);
				wasm_v128_store( dst + ( p + 0 ) * 8 + 0, c0 );
				wasm_v128_store( dst + ( p + 0 ) * 8 + 4, d0 );
				wasm_v128_store( dst + ( p + 1 ) * 8 + 0, c1 );
				wasm_v128_store( dst + ( p + 1 ) * 8 + 4, d1 );
				wasm_v128_store( dst + ( p + 2 ) * 8 + 0, c2 );
				wasm_v128_store( dst + ( p + 2 ) * 8 + 4, d2 );
				wasm_v128_store( dst + ( p + 3 ) * 8 + 0, c3 );
				wasm_v128_store( dst + ( p + 3 ) * 8 + 4, d3 );
			}
			for ( ; p < k; ++p )
			{
				float* out = dst + p * 8;
				out[0] = b0[p]; out[1] = b1[p]; out[2] = b2[p]; out[3] = b3[p];
				out[4] = b4[p]; out[5] = b5[p]; out[6] = b6[p]; out[7] = b7[p];
			}
			return;
		}
	}

	if ( nr <= 4 )
	{
		// Pack with stride_b = 4 for ukernel_4x4_packed!
		if ( cs_b == 1 && nr == 4 )
		{
			for ( dim_t p = 0; p < k; ++p )
				wasm_v128_store( dst + p * 4, wasm_v128_load( b + p * rs_b ) );
			return;
		}
		if ( rs_b == 1 )
		{
			const v128_t vzero = wasm_f32x4_const_splat( 0.0f );
			const float* b0 = b;
			const float* b1 = nr >= 2 ? b + 1 * cs_b : NULL;
			const float* b2 = nr >= 3 ? b + 2 * cs_b : NULL;
			const float* b3 = nr >= 4 ? b + 3 * cs_b : NULL;
			dim_t p = 0;
			for ( ; p + 4 <= k; p += 4 )
			{
				v128_t c0, c1, c2, c3;
				bli_sgemmsup_wasm32_transpose_4x4
				(
				  wasm_v128_load( b0 + p ),
				  b1 ? wasm_v128_load( b1 + p ) : vzero,
				  b2 ? wasm_v128_load( b2 + p ) : vzero,
				  b3 ? wasm_v128_load( b3 + p ) : vzero,
				  &c0, &c1, &c2, &c3
				);
				wasm_v128_store( dst + ( p + 0 ) * 4, c0 );
				wasm_v128_store( dst + ( p + 1 ) * 4, c1 );
				wasm_v128_store( dst + ( p + 2 ) * 4, c2 );
				wasm_v128_store( dst + ( p + 3 ) * 4, c3 );
			}
			for ( ; p < k; ++p )
			{
				dst[p * 4 + 0] = b0[p];
				dst[p * 4 + 1] = b1 ? b1[p] : 0.0f;
				dst[p * 4 + 2] = b2 ? b2[p] : 0.0f;
				dst[p * 4 + 3] = b3 ? b3[p] : 0.0f;
			}
			return;
		}
		for ( dim_t p = 0; p < k; ++p )
		{
			const float* src = b + p * rs_b;
			float*       out = dst + p * 4;
			out[0] = src[0];
			out[1] = nr >= 2 ? src[1 * cs_b] : 0.0f;
			out[2] = nr >= 3 ? src[2 * cs_b] : 0.0f;
			out[3] = 0.0f;
		}
		return;
	}

	// 5 <= nr < 8: pack with stride_b = 8 for ukernel_4x8_packed
	if ( rs_b == 1 )
	{
		const v128_t vzero = wasm_f32x4_const_splat( 0.0f );
		const float* b0 = b + 0 * cs_b;
		const float* b1 = b + 1 * cs_b;
		const float* b2 = b + 2 * cs_b;
		const float* b3 = b + 3 * cs_b;
		const float* b4 = b + 4 * cs_b;
		const float* b5 = nr >= 6 ? b + 5 * cs_b : NULL;
		const float* b6 = nr >= 7 ? b + 6 * cs_b : NULL;
		dim_t p = 0;
		for ( ; p + 4 <= k; p += 4 )
		{
			v128_t c0, c1, c2, c3, d0, d1, d2, d3;
			bli_sgemmsup_wasm32_transpose_4x4
			(
			  wasm_v128_load( b0 + p ),
			  wasm_v128_load( b1 + p ),
			  wasm_v128_load( b2 + p ),
			  wasm_v128_load( b3 + p ),
			  &c0, &c1, &c2, &c3
			);
			bli_sgemmsup_wasm32_transpose_4x4
			(
			  wasm_v128_load( b4 + p ),
			  b5 ? wasm_v128_load( b5 + p ) : vzero,
			  b6 ? wasm_v128_load( b6 + p ) : vzero,
			  vzero,
			  &d0, &d1, &d2, &d3
			);
			wasm_v128_store( dst + ( p + 0 ) * 8 + 0, c0 );
			wasm_v128_store( dst + ( p + 0 ) * 8 + 4, d0 );
			wasm_v128_store( dst + ( p + 1 ) * 8 + 0, c1 );
			wasm_v128_store( dst + ( p + 1 ) * 8 + 4, d1 );
			wasm_v128_store( dst + ( p + 2 ) * 8 + 0, c2 );
			wasm_v128_store( dst + ( p + 2 ) * 8 + 4, d2 );
			wasm_v128_store( dst + ( p + 3 ) * 8 + 0, c3 );
			wasm_v128_store( dst + ( p + 3 ) * 8 + 4, d3 );
		}
		for ( ; p < k; ++p )
		{
			float* out = dst + p * 8;
			out[0] = b0[p]; out[1] = b1[p]; out[2] = b2[p]; out[3] = b3[p];
			out[4] = b4[p];
			out[5] = b5 ? b5[p] : 0.0f;
			out[6] = b6 ? b6[p] : 0.0f;
			out[7] = 0.0f;
		}
		return;
	}

	for ( dim_t p = 0; p < k; ++p )
	{
		const float* src = b + p * rs_b;
		float*       out = dst + p * 8;
		wasm_v128_store( out + 0, wasm_v128_load( src + 0 ) );
		out[4] = src[4];
		out[5] = nr >= 6 ? src[5] : 0.0f;
		out[6] = nr >= 7 ? src[6] : 0.0f;
		out[7] = 0.0f;
	}
}

// Packing buffers shared by the two compute paths. They are static rather
// than automatic so that the handler keeps a small stack frame: together
// they take 36 KiB, and Emscripten's default stack is 64 KiB with no
// overflow check in optimized builds. The handler does not call itself, and
// BLIS_THREAD_LOCAL gives every thread its own copy unless TLS is disabled
// at configure time (which BLIS documents as unsafe for concurrent calls).
static BLIS_THREAD_LOCAL float bli_sgemmsup_wasm32_pack_big[ 64 * 128 ]
       __attribute__((aligned(16)));
static BLIS_THREAD_LOCAL float bli_sgemmsup_wasm32_pack_sm[ 8 * 128 ]
       __attribute__((aligned(16)));

// Path for m, n, k <= 64: B is packed once into 8-wide panels and each
// 4-row panel of A is packed in turn.
__attribute__((noinline))
static void bli_sgemmsup_wasm32_compute_64
     (
       dim_t        m,
       dim_t        n,
       dim_t        k,
       float        alpha,
       const float* a,
       inc_t        rs_a,
       inc_t        cs_a,
       const float* b,
       inc_t        rs_b,
       inc_t        cs_b,
       float        beta,
       float*       c,
       inc_t        rs_c
     )
{
	float* pack_b = bli_sgemmsup_wasm32_pack_big; // uses up to 64 * 64
	float* pack_a = bli_sgemmsup_wasm32_pack_sm;  // uses up to  4 * 64

	const dim_t n_panels = ( n + 7 ) / 8;
	for ( dim_t jp = 0; jp < n_panels; ++jp )
	{
		const dim_t j  = jp * 8;
		const dim_t nr = ( n - j < 8 ) ? ( n - j ) : 8;
		bli_sgemmsup_wasm32_pack_b_panel
		(
		  nr, k,
		  b + j * cs_b, rs_b, cs_b,
		  pack_b + jp * 8 * k
		);
	}

	for ( dim_t i = 0; i < m; i += 4 )
	{
		const dim_t mr = ( m - i < 4 ) ? ( m - i ) : 4;
		bli_sgemmsup_wasm32_pack_a_panel
		(
		  mr, k,
		  a + i * rs_a, rs_a, cs_a,
		  pack_a
		);

		for ( dim_t jp = 0; jp < n_panels; ++jp )
		{
			const dim_t  j   = jp * 8;
			const dim_t  nr  = ( n - j < 8 ) ? ( n - j ) : 8;
			const float* b_p = pack_b + jp * 8 * k;
			float*       c_p = c + i * rs_c + j;

			if ( nr > 4 )
			{
				bli_sgemmsup_wasm32_ukernel_4x8_packed
				(
				  mr, nr, k, alpha,
				  pack_a, b_p,
				  beta, c_p, rs_c
				);
			}
			else
			{
				bli_sgemmsup_wasm32_ukernel_4x4_packed
				(
				  mr, nr, k, alpha,
				  pack_a, b_p,
				  beta, c_p, rs_c
				);
			}
		}
	}
}

// Path for 64 < max(m, n, k) <= 128: depending on the operands' storage,
// either B is packed in blocks of up to 64 columns and A one 4-row panel at
// a time, or A in blocks of up to 64 rows and B one 8-column panel at a
// time.
__attribute__((noinline))
static void bli_sgemmsup_wasm32_compute_128
     (
       dim_t        m,
       dim_t        n,
       dim_t        k,
       float        alpha,
       const float* a,
       inc_t        rs_a,
       inc_t        cs_a,
       const float* b,
       inc_t        rs_b,
       inc_t        cs_b,
       float        beta,
       float*       c,
       inc_t        rs_c
     )
{
	float* pack_big = bli_sgemmsup_wasm32_pack_big; // uses up to 64 * 128
	float* pack_sm  = bli_sgemmsup_wasm32_pack_sm;  // uses up to  8 * 128

	if ( rs_b == 1 && cs_a == 1 )
	{
		for ( dim_t j0 = 0; j0 < n; j0 += 64 )
		{
			const dim_t nc = ( n - j0 < 64 ) ? ( n - j0 ) : 64;
			const dim_t n_panels = ( nc + 7 ) / 8;
			for ( dim_t jp = 0; jp < n_panels; ++jp )
			{
				const dim_t j  = jp * 8;
				const dim_t nr = ( nc - j < 8 ) ? ( nc - j ) : 8;
				bli_sgemmsup_wasm32_pack_b_panel
				(
				  nr, k,
				  b + ( j0 + j ) * cs_b, rs_b, cs_b,
				  pack_big + jp * 8 * k
				);
			}

			for ( dim_t i = 0; i < m; i += 4 )
			{
				const dim_t mr = ( m - i < 4 ) ? ( m - i ) : 4;
				bli_sgemmsup_wasm32_pack_a_panel
				(
				  mr, k,
				  a + i * rs_a, rs_a, cs_a,
				  pack_sm
				);

				for ( dim_t jp = 0; jp < n_panels; ++jp )
				{
					const dim_t  j   = jp * 8;
					const dim_t  nr  = ( nc - j < 8 ) ? ( nc - j ) : 8;
					const float* b_p = pack_big + jp * 8 * k;
					float*       c_p = c + i * rs_c + ( j0 + j );

					if ( nr > 4 )
					{
						bli_sgemmsup_wasm32_ukernel_4x8_packed
						(
						  mr, nr, k, alpha,
						  pack_sm, b_p,
						  beta, c_p, rs_c
						);
					}
					else
					{
						bli_sgemmsup_wasm32_ukernel_4x4_packed
						(
						  mr, nr, k, alpha,
						  pack_sm, b_p,
						  beta, c_p, rs_c
						);
					}
				}
			}
		}
	}
	else
	{
		for ( dim_t i0 = 0; i0 < m; i0 += 64 )
		{
			const dim_t mc = ( m - i0 < 64 ) ? ( m - i0 ) : 64;
			const dim_t m_panels = ( mc + 3 ) / 4;
			for ( dim_t ip = 0; ip < m_panels; ++ip )
			{
				const dim_t i  = ip * 4;
				const dim_t mr = ( mc - i < 4 ) ? ( mc - i ) : 4;
				bli_sgemmsup_wasm32_pack_a_panel
				(
				  mr, k,
				  a + ( i0 + i ) * rs_a, rs_a, cs_a,
				  pack_big + ip * 4 * k
				);
			}

			for ( dim_t j = 0; j < n; j += 8 )
			{
				const dim_t nr = ( n - j < 8 ) ? ( n - j ) : 8;
				bli_sgemmsup_wasm32_pack_b_panel
				(
				  nr, k,
				  b + j * cs_b, rs_b, cs_b,
				  pack_sm
				);

				for ( dim_t ip = 0; ip < m_panels; ++ip )
				{
					const dim_t  i   = ip * 4;
					const dim_t  mr  = ( mc - i < 4 ) ? ( mc - i ) : 4;
					const float* a_p = pack_big + ip * 4 * k;
					float*       c_p = c + ( i0 + i ) * rs_c + j;

					if ( nr > 4 )
					{
						bli_sgemmsup_wasm32_ukernel_4x8_packed
						(
						  mr, nr, k, alpha,
						  a_p, pack_sm,
						  beta, c_p, rs_c
						);
					}
					else
					{
						bli_sgemmsup_wasm32_ukernel_4x4_packed
						(
						  mr, nr, k, alpha,
						  a_p, pack_sm,
						  beta, c_p, rs_c
						);
					}
				}
			}
		}
	}
}

static inline void bli_sgemmsup_wasm32_dispatch
     (
       dim_t        m,
       dim_t        n,
       dim_t        k,
       float        alpha,
       const float* a,
       inc_t        rs_a,
       inc_t        cs_a,
       const float* b,
       inc_t        rs_b,
       inc_t        cs_b,
       float        beta,
       float*       c,
       inc_t        rs_c,
       inc_t        cs_c
     )
{
	( void )cs_c;
	if ( m <= 4 && n == 8 )
	{
		bli_sgemmsup_wasm32_direct_4x8( m, 8, k, alpha, a, rs_a, cs_a, b, rs_b, cs_b, beta, c, rs_c );
		return;
	}
	if ( m == 8 && n == 4 )
	{
		bli_sgemmsup_wasm32_direct_8x4( k, alpha, a, rs_a, cs_a, b, rs_b, cs_b, beta, c, rs_c );
		return;
	}
	if ( m <= 64 && n <= 64 && k <= 64 )
	{
		bli_sgemmsup_wasm32_compute_64( m, n, k, alpha, a, rs_a, cs_a, b, rs_b, cs_b, beta, c, rs_c );
	}
	else
	{
		bli_sgemmsup_wasm32_compute_128( m, n, k, alpha, a, rs_a, cs_a, b, rs_b, cs_b, beta, c, rs_c );
	}
}

static inline bool bli_sgemmsup_wasm32_cntx_is_default
     (
       const cntx_t* cntx,
       dim_t         m,
       dim_t         n,
       dim_t         k
     )
{
	if ( !bli_cntx_l3_sup_thresh_is_met( BLIS_FLOAT, m, n, k, cntx ) ) return false;
	if ( bli_cntx_get_l3_sup_handler( BLIS_GEMM, cntx ) !=
	     ( void_fp )bli_sgemmsup_wasm32_simd128 ) return false;
	if ( bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_MR, cntx ) != 4 ) return false;
	if ( bli_cntx_get_blksz_max_dt( BLIS_FLOAT, BLIS_MR, cntx ) != 4 ) return false;
	if ( bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_NR, cntx ) != 8 ) return false;
	if ( bli_cntx_get_blksz_max_dt( BLIS_FLOAT, BLIS_NR, cntx ) != 8 ) return false;
	if ( bli_cntx_get_blksz_max_dt( BLIS_FLOAT, BLIS_BBM, cntx ) != 1 ) return false;
	if ( bli_cntx_get_blksz_max_dt( BLIS_FLOAT, BLIS_BBN, cntx ) != 1 ) return false;
	if ( !bli_cntx_get_ukr_prefs_dt( BLIS_FLOAT, BLIS_GEMM_UKR_ROW_PREF, cntx ) ) return false;
	if ( bli_cntx_get_ukr2_dt( BLIS_FLOAT, BLIS_FLOAT, BLIS_GEMM_UKR, cntx ) !=
	     ( void_fp )bli_sgemm_wasm32_simd128_4x8 ) return false;
	if ( bli_cntx_get_ukr2_dt( BLIS_FLOAT, BLIS_FLOAT, BLIS_PACKM_KER, cntx ) !=
	     ( void_fp )bli_sspackm_wasm32_ref ) return false;
	return true;
}

err_t bli_sgemmsup_wasm32_simd128
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       rntm_t*       rntm
     )
{
#ifdef BLIS_DISABLE_SUP_HANDLING
	return BLIS_FAILURE;
#endif

	if ( cntx == NULL || rntm == NULL || a == NULL || b == NULL || c == NULL )
		return BLIS_FAILURE;
	if ( !bli_sgemmsup_wasm32_rntm_is_single_thread( rntm ) ) return BLIS_FAILURE;
	if ( !bli_sgemmsup_wasm32_obj_ok( a ) ||
	     !bli_sgemmsup_wasm32_obj_ok( b ) ||
	     !bli_sgemmsup_wasm32_obj_ok( c ) ) return BLIS_FAILURE;

	dim_t m = bli_obj_length_after_trans( c );
	dim_t n = bli_obj_width_after_trans( c );
	const dim_t k = bli_obj_width_after_trans( a );
	if ( m <= 0 || n <= 0 || k <= 0 ) return BLIS_FAILURE;
	if ( m > 128 || n > 128 || k > 128 ) return BLIS_FAILURE;
	if ( bli_obj_length_after_trans( a ) != m ||
	     bli_obj_length_after_trans( b ) != k ||
	     bli_obj_width_after_trans( b ) != n ) return BLIS_FAILURE;

	float alpha_val = 1.0f;
	float beta_val  = 0.0f;
	if ( !bli_sgemmsup_wasm32_extract_scalar( alpha, &alpha_val ) ||
	     !bli_sgemmsup_wasm32_extract_scalar( beta,  &beta_val ) )
		return BLIS_FAILURE;
	if ( alpha_val == 0.0f ) return BLIS_FAILURE;

	const inc_t rs_c = bli_obj_has_trans( c ) ? bli_obj_col_stride( c ) : bli_obj_row_stride( c );
	const inc_t cs_c = bli_obj_has_trans( c ) ? bli_obj_row_stride( c ) : bli_obj_col_stride( c );
	const bool row_pref = bli_cntx_get_ukr_prefs_dt( BLIS_FLOAT, BLIS_GEMM_UKR_ROW_PREF, cntx );
	const bool is_col = bli_sgemmsup_wasm32_abs( rs_c ) == 1 && bli_sgemmsup_wasm32_abs( cs_c ) != 1;
	const bool is_row = bli_sgemmsup_wasm32_abs( cs_c ) == 1 && bli_sgemmsup_wasm32_abs( rs_c ) != 1;

	const dim_t mt = ( ( row_pref && is_col ) || ( !row_pref && is_row ) ) ? n : m;
	const dim_t nt = ( ( row_pref && is_col ) || ( !row_pref && is_row ) ) ? m : n;
	if ( !bli_sgemmsup_wasm32_cntx_is_default( cntx, mt, nt, k ) ) return BLIS_FAILURE;

	const float* a_buf;
	const float* b_buf;
	const float* c_buf_const;
	inc_t rs_a0, cs_a0, rs_b0, cs_b0, rs_c0, cs_c0;

	bli_sgemmsup_wasm32_extract_view( a, &a_buf, &rs_a0, &cs_a0 );
	bli_sgemmsup_wasm32_extract_view( b, &b_buf, &rs_b0, &cs_b0 );
	bli_sgemmsup_wasm32_extract_view( c, &c_buf_const, &rs_c0, &cs_c0 );

	if ( ( row_pref && is_col ) || ( !row_pref && is_row ) )
	{
		const dim_t tmp_d = m; m = n; n = tmp_d;
		const float* tmp_p = a_buf; a_buf = b_buf; b_buf = tmp_p;
		inc_t tmp_s;
		tmp_s = rs_a0; rs_a0 = cs_b0; cs_b0 = tmp_s;
		tmp_s = cs_a0; cs_a0 = rs_b0; rs_b0 = tmp_s;
		tmp_s = rs_c0; rs_c0 = cs_c0; cs_c0 = tmp_s;
	}

	if ( cs_c0 != 1 ) return BLIS_FAILURE;
	if ( !bli_sgemmsup_wasm32_storage_ok( m, k, rs_a0, cs_a0 ) ||
	     !bli_sgemmsup_wasm32_storage_ok( k, n, rs_b0, cs_b0 ) ||
	     !bli_sgemmsup_wasm32_storage_ok( m, n, rs_c0, cs_c0 ) )
		return BLIS_FAILURE;

	bli_sgemmsup_wasm32_dispatch
	(
	  m, n, k,
	  alpha_val,
	  a_buf, rs_a0, cs_a0,
	  b_buf, rs_b0, cs_b0,
	  beta_val,
	  ( float* )c_buf_const, rs_c0, cs_c0
	);

	return BLIS_SUCCESS;
}

// The float MT threshold above is shared by GEMM and GEMMT. Without this
// wrapper, the reference GEMMT sup handler would start taking small float
// GEMMT problems. Float GEMMT therefore stays on the conventional path, and
// every other datatype keeps bli_gemmtsup_ref() with its unchanged
// thresholds.
err_t bli_gemmtsup_wasm32
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             rntm_t* rntm
     )
{
	if ( bli_obj_dt( c ) == BLIS_FLOAT ) return BLIS_FAILURE;
	return bli_gemmtsup_ref( alpha, a, b, beta, c, cntx, rntm );
}
