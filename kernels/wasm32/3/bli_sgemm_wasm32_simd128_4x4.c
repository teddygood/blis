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

void bli_sgemm_wasm32_simd128_4x4
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
	GEMM_UKR_SETUP_CT( s, 4, 4, true );

	v128_t c0v = wasm_f32x4_splat( 0.0f );
	v128_t c1v = wasm_f32x4_splat( 0.0f );
	v128_t c2v = wasm_f32x4_splat( 0.0f );
	v128_t c3v = wasm_f32x4_splat( 0.0f );

	for ( dim_t p = 0; p < k; ++p )
	{
		const v128_t bv = wasm_v128_load( b );
		c0v = wasm_f32x4_add( c0v, wasm_f32x4_mul( wasm_f32x4_splat( a[0] ), bv ) );
		c1v = wasm_f32x4_add( c1v, wasm_f32x4_mul( wasm_f32x4_splat( a[1] ), bv ) );
		c2v = wasm_f32x4_add( c2v, wasm_f32x4_mul( wasm_f32x4_splat( a[2] ), bv ) );
		c3v = wasm_f32x4_add( c3v, wasm_f32x4_mul( wasm_f32x4_splat( a[3] ), bv ) );
		a += 4;
		b += 4;
	}

	const v128_t av = wasm_f32x4_splat( *alpha );
	c0v = wasm_f32x4_mul( av, c0v );
	c1v = wasm_f32x4_mul( av, c1v );
	c2v = wasm_f32x4_mul( av, c2v );
	c3v = wasm_f32x4_mul( av, c3v );

	if ( !bli_teq0s( s, *beta ) )
	{
		const v128_t bv = wasm_f32x4_splat( *beta );
		c0v = wasm_f32x4_add( c0v, wasm_f32x4_mul( bv, wasm_v128_load( c ) ) );
		c1v = wasm_f32x4_add( c1v, wasm_f32x4_mul( bv, wasm_v128_load( c + rs_c ) ) );
		c2v = wasm_f32x4_add( c2v, wasm_f32x4_mul( bv, wasm_v128_load( c + 2*rs_c ) ) );
		c3v = wasm_f32x4_add( c3v, wasm_f32x4_mul( bv, wasm_v128_load( c + 3*rs_c ) ) );
	}

	wasm_v128_store( c,          c0v );
	wasm_v128_store( c + rs_c,   c1v );
	wasm_v128_store( c + 2*rs_c, c2v );
	wasm_v128_store( c + 3*rs_c, c3v );

	GEMM_UKR_FLUSH_CT( s );
}
