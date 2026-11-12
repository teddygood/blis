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

void bli_sgemm_wasm32_simd128_4x8
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha0,
       const void*      a0,
       const void*      b0,
       const void*      beta0,
             void*      c0, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const float* alpha = alpha0;
	const float* a     = a0;
	const float* b     = b0;
	const float* beta  = beta0;
	      float* c     = c0;

	( void )data;
	( void )cntx;

	if ( m == 0 || n == 0 ) return;

	// Use a temporary tile for edges and noncontiguous output.
	GEMM_UKR_SETUP_CT( s, 4, 8, true );

	v128_t c00v = wasm_f32x4_splat( 0.0f );
	v128_t c01v = wasm_f32x4_splat( 0.0f );
	v128_t c10v = wasm_f32x4_splat( 0.0f );
	v128_t c11v = wasm_f32x4_splat( 0.0f );
	v128_t c20v = wasm_f32x4_splat( 0.0f );
	v128_t c21v = wasm_f32x4_splat( 0.0f );
	v128_t c30v = wasm_f32x4_splat( 0.0f );
	v128_t c31v = wasm_f32x4_splat( 0.0f );

	for ( dim_t p = 0; p < k; ++p )
	{
		const v128_t b0v = wasm_v128_load( b );
		const v128_t b1v = wasm_v128_load( b + 4 );
		const v128_t a0v = wasm_f32x4_splat( a[0] );
		const v128_t a1v = wasm_f32x4_splat( a[1] );
		const v128_t a2v = wasm_f32x4_splat( a[2] );
		const v128_t a3v = wasm_f32x4_splat( a[3] );

		// Reuse each broadcast across two adjacent vectors of a row.
		c00v = wasm_f32x4_add( c00v, wasm_f32x4_mul( a0v, b0v ) );
		c01v = wasm_f32x4_add( c01v, wasm_f32x4_mul( a0v, b1v ) );
		c10v = wasm_f32x4_add( c10v, wasm_f32x4_mul( a1v, b0v ) );
		c11v = wasm_f32x4_add( c11v, wasm_f32x4_mul( a1v, b1v ) );
		c20v = wasm_f32x4_add( c20v, wasm_f32x4_mul( a2v, b0v ) );
		c21v = wasm_f32x4_add( c21v, wasm_f32x4_mul( a2v, b1v ) );
		c30v = wasm_f32x4_add( c30v, wasm_f32x4_mul( a3v, b0v ) );
		c31v = wasm_f32x4_add( c31v, wasm_f32x4_mul( a3v, b1v ) );
		a += 4;
		b += 8;
	}

	const v128_t av = wasm_f32x4_splat( *alpha );
	c00v = wasm_f32x4_mul( av, c00v );
	c01v = wasm_f32x4_mul( av, c01v );
	c10v = wasm_f32x4_mul( av, c10v );
	c11v = wasm_f32x4_mul( av, c11v );
	c20v = wasm_f32x4_mul( av, c20v );
	c21v = wasm_f32x4_mul( av, c21v );
	c30v = wasm_f32x4_mul( av, c30v );
	c31v = wasm_f32x4_mul( av, c31v );

	if ( !bli_teq0s( s, *beta ) )
	{
		const v128_t bv = wasm_f32x4_splat( *beta );
		c00v = wasm_f32x4_add( c00v, wasm_f32x4_mul( bv, wasm_v128_load( c ) ) );
		c01v = wasm_f32x4_add( c01v, wasm_f32x4_mul( bv, wasm_v128_load( c + 4 ) ) );
		c10v = wasm_f32x4_add( c10v, wasm_f32x4_mul( bv, wasm_v128_load( c + rs_c ) ) );
		c11v = wasm_f32x4_add( c11v, wasm_f32x4_mul( bv, wasm_v128_load( c + rs_c + 4 ) ) );
		c20v = wasm_f32x4_add( c20v, wasm_f32x4_mul( bv, wasm_v128_load( c + 2*rs_c ) ) );
		c21v = wasm_f32x4_add( c21v, wasm_f32x4_mul( bv, wasm_v128_load( c + 2*rs_c + 4 ) ) );
		c30v = wasm_f32x4_add( c30v, wasm_f32x4_mul( bv, wasm_v128_load( c + 3*rs_c ) ) );
		c31v = wasm_f32x4_add( c31v, wasm_f32x4_mul( bv, wasm_v128_load( c + 3*rs_c + 4 ) ) );
	}

	wasm_v128_store( c,              c00v );
	wasm_v128_store( c + 4,          c01v );
	wasm_v128_store( c + rs_c,       c10v );
	wasm_v128_store( c + rs_c + 4,   c11v );
	wasm_v128_store( c + 2*rs_c,     c20v );
	wasm_v128_store( c + 2*rs_c + 4, c21v );
	wasm_v128_store( c + 3*rs_c,     c30v );
	wasm_v128_store( c + 3*rs_c + 4, c31v );

	GEMM_UKR_FLUSH_CT( s );
}
