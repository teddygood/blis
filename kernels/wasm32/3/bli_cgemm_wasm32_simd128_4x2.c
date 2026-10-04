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

// Rotate each complex pair from (real, imag) to (-imag, real).
static inline v128_t rotate( v128_t x )
{
	return wasm_v128_xor( wasm_i32x4_shuffle( x, x, 1, 0, 3, 2 ),
	                      wasm_i32x4_make( 0x80000000u, 0, 0x80000000u, 0 ) );
}

static inline v128_t scale( v128_t x, scomplex a )
{
	return wasm_f32x4_add( wasm_f32x4_mul( wasm_f32x4_splat( a.real ), x ),
	                       wasm_f32x4_mul( wasm_f32x4_splat( a.imag ), rotate( x ) ) );
}

void bli_cgemm_wasm32_simd128_4x2
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
	const scomplex* alpha = alpha0;
	const scomplex* a     = a0;
	const scomplex* b     = b0;
	const scomplex* beta  = beta0;
	      scomplex* c     = c0;

	( void )data;
	( void )cntx;

	if ( m == 0 || n == 0 ) return;

	GEMM_UKR_SETUP_CT( c, 4, 2, true );

	v128_t c00 = wasm_f32x4_splat( 0 );
	v128_t c10 = wasm_f32x4_splat( 0 );
	v128_t c20 = wasm_f32x4_splat( 0 );
	v128_t c30 = wasm_f32x4_splat( 0 );

	for ( dim_t p = 0; p < k; ++p )
	{
		const v128_t b0 = wasm_v128_load( b );
		const v128_t r0 = rotate( b0 );
		const v128_t a0r = wasm_f32x4_splat( a[0].real );
		const v128_t a0i = wasm_f32x4_splat( a[0].imag );
		c00 = wasm_f32x4_add( c00, wasm_f32x4_add(
		        wasm_f32x4_mul( a0r, b0 ), wasm_f32x4_mul( a0i, r0 ) ) );
		const v128_t a1r = wasm_f32x4_splat( a[1].real );
		const v128_t a1i = wasm_f32x4_splat( a[1].imag );
		c10 = wasm_f32x4_add( c10, wasm_f32x4_add(
		        wasm_f32x4_mul( a1r, b0 ), wasm_f32x4_mul( a1i, r0 ) ) );
		const v128_t a2r = wasm_f32x4_splat( a[2].real );
		const v128_t a2i = wasm_f32x4_splat( a[2].imag );
		c20 = wasm_f32x4_add( c20, wasm_f32x4_add(
		        wasm_f32x4_mul( a2r, b0 ), wasm_f32x4_mul( a2i, r0 ) ) );
		const v128_t a3r = wasm_f32x4_splat( a[3].real );
		const v128_t a3i = wasm_f32x4_splat( a[3].imag );
		c30 = wasm_f32x4_add( c30, wasm_f32x4_add(
		        wasm_f32x4_mul( a3r, b0 ), wasm_f32x4_mul( a3i, r0 ) ) );
		a += 4;
		b += 2;
	}

	c00 = scale( c00, *alpha );
	c10 = scale( c10, *alpha );
	c20 = scale( c20, *alpha );
	c30 = scale( c30, *alpha );

	if ( !bli_teq0s( c, *beta ) )
	{
		c00 = wasm_f32x4_add( c00, scale( wasm_v128_load( c ), *beta ) );
		c10 = wasm_f32x4_add( c10, scale( wasm_v128_load( c + rs_c ), *beta ) );
		c20 = wasm_f32x4_add( c20, scale( wasm_v128_load( c + 2*rs_c ), *beta ) );
		c30 = wasm_f32x4_add( c30, scale( wasm_v128_load( c + 3*rs_c ), *beta ) );
	}

	wasm_v128_store( c, c00 );
	wasm_v128_store( c + rs_c, c10 );
	wasm_v128_store( c + 2*rs_c, c20 );
	wasm_v128_store( c + 3*rs_c, c30 );

	GEMM_UKR_FLUSH_CT( c );
}
