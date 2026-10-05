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
	return wasm_v128_xor( wasm_i64x2_shuffle( x, x, 1, 0 ),
	                      wasm_i64x2_make( 0x8000000000000000ULL, 0 ) );
}

static inline v128_t scale( v128_t x, dcomplex a )
{
	return wasm_f64x2_add( wasm_f64x2_mul( wasm_f64x2_splat( a.real ), x ),
	                       wasm_f64x2_mul( wasm_f64x2_splat( a.imag ), rotate( x ) ) );
}

void bli_zgemm_wasm32_simd128_2x2
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
	const dcomplex* alpha = alpha0;
	const dcomplex* a     = a0;
	const dcomplex* b     = b0;
	const dcomplex* beta  = beta0;
	      dcomplex* c     = c0;

	( void )data;
	( void )cntx;

	if ( m == 0 || n == 0 ) return;

	GEMM_UKR_SETUP_CT( z, 2, 2, true );

	v128_t c00 = wasm_f64x2_splat( 0 );
	v128_t c01 = wasm_f64x2_splat( 0 );
	v128_t c10 = wasm_f64x2_splat( 0 );
	v128_t c11 = wasm_f64x2_splat( 0 );

	for ( dim_t p = 0; p < k; ++p )
	{
		const v128_t b0 = wasm_v128_load( b );
		const v128_t r0 = rotate( b0 );
		const v128_t b1 = wasm_v128_load( b + 1 );
		const v128_t r1 = rotate( b1 );
		const v128_t a0r = wasm_f64x2_splat( a[0].real );
		const v128_t a0i = wasm_f64x2_splat( a[0].imag );
		c00 = wasm_f64x2_add( c00, wasm_f64x2_add(
		        wasm_f64x2_mul( a0r, b0 ), wasm_f64x2_mul( a0i, r0 ) ) );
		c01 = wasm_f64x2_add( c01, wasm_f64x2_add(
		        wasm_f64x2_mul( a0r, b1 ), wasm_f64x2_mul( a0i, r1 ) ) );
		const v128_t a1r = wasm_f64x2_splat( a[1].real );
		const v128_t a1i = wasm_f64x2_splat( a[1].imag );
		c10 = wasm_f64x2_add( c10, wasm_f64x2_add(
		        wasm_f64x2_mul( a1r, b0 ), wasm_f64x2_mul( a1i, r0 ) ) );
		c11 = wasm_f64x2_add( c11, wasm_f64x2_add(
		        wasm_f64x2_mul( a1r, b1 ), wasm_f64x2_mul( a1i, r1 ) ) );
		a += 2;
		b += 2;
	}

	c00 = scale( c00, *alpha );
	c01 = scale( c01, *alpha );
	c10 = scale( c10, *alpha );
	c11 = scale( c11, *alpha );

	if ( !bli_teq0s( z, *beta ) )
	{
		c00 = wasm_f64x2_add( c00, scale( wasm_v128_load( c ), *beta ) );
		c01 = wasm_f64x2_add( c01, scale( wasm_v128_load( c + 1 ), *beta ) );
		c10 = wasm_f64x2_add( c10, scale( wasm_v128_load( c + rs_c ), *beta ) );
		c11 = wasm_f64x2_add( c11, scale( wasm_v128_load( c + rs_c + 1 ), *beta ) );
	}

	wasm_v128_store( c, c00 );
	wasm_v128_store( c + 1, c01 );
	wasm_v128_store( c + rs_c, c10 );
	wasm_v128_store( c + rs_c + 1, c11 );

	GEMM_UKR_FLUSH_CT( z );
}
