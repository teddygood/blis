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

void bli_dgemm_wasm32_simd128_4x4
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
	const double* alpha = alpha0;
	const double* a     = a0;
	const double* b     = b0;
	const double* beta  = beta0;
	      double* c     = c0;

	( void )data;
	( void )cntx;

	if ( m == 0 || n == 0 ) return;

	// Redirect column-stored, general-stored, and partial tiles to a
	// row-contiguous temporary microtile; beta is then reset to zero and
	// the tile is flushed back to c with the original beta below.
	GEMM_UKR_SETUP_CT( d, 4, 4, true );

	// Eight accumulators: row i holds columns 0/1 in ci0 and 2/3 in ci1.
	v128_t c00 = wasm_f64x2_splat( 0.0 );
	v128_t c01 = wasm_f64x2_splat( 0.0 );
	v128_t c10 = wasm_f64x2_splat( 0.0 );
	v128_t c11 = wasm_f64x2_splat( 0.0 );
	v128_t c20 = wasm_f64x2_splat( 0.0 );
	v128_t c21 = wasm_f64x2_splat( 0.0 );
	v128_t c30 = wasm_f64x2_splat( 0.0 );
	v128_t c31 = wasm_f64x2_splat( 0.0 );

	// Packed A holds mr=4 contiguous scalars per k-slice; packed B holds
	// nr=4 contiguous scalars per k-slice. BLIS pads partial panels, so a
	// full 4x4 update is valid for every nonempty m,n.
	for ( dim_t p = 0; p < k; ++p )
	{
		const v128_t b01 = wasm_v128_load( b );
		const v128_t b23 = wasm_v128_load( b + 2 );

		const v128_t av0 = wasm_f64x2_splat( a[0] );
		const v128_t av1 = wasm_f64x2_splat( a[1] );
		const v128_t av2 = wasm_f64x2_splat( a[2] );
		const v128_t av3 = wasm_f64x2_splat( a[3] );

		c00 = wasm_f64x2_add( c00, wasm_f64x2_mul( av0, b01 ) );
		c01 = wasm_f64x2_add( c01, wasm_f64x2_mul( av0, b23 ) );
		c10 = wasm_f64x2_add( c10, wasm_f64x2_mul( av1, b01 ) );
		c11 = wasm_f64x2_add( c11, wasm_f64x2_mul( av1, b23 ) );
		c20 = wasm_f64x2_add( c20, wasm_f64x2_mul( av2, b01 ) );
		c21 = wasm_f64x2_add( c21, wasm_f64x2_mul( av2, b23 ) );
		c30 = wasm_f64x2_add( c30, wasm_f64x2_mul( av3, b01 ) );
		c31 = wasm_f64x2_add( c31, wasm_f64x2_mul( av3, b23 ) );

		a += 4;
		b += 4;
	}

	// Scale the finished product by alpha, matching the reference order.
	const v128_t alphav = wasm_f64x2_splat( *alpha );
	c00 = wasm_f64x2_mul( c00, alphav );
	c01 = wasm_f64x2_mul( c01, alphav );
	c10 = wasm_f64x2_mul( c10, alphav );
	c11 = wasm_f64x2_mul( c11, alphav );
	c20 = wasm_f64x2_mul( c20, alphav );
	c21 = wasm_f64x2_mul( c21, alphav );
	c30 = wasm_f64x2_mul( c30, alphav );
	c31 = wasm_f64x2_mul( c31, alphav );

	// Only row-contiguous c reaches this point (cs_c == 1), so each row is
	// two aligned-or-unaligned 16-byte segments.
	if ( bli_teq0s( d, *beta ) )
	{
		wasm_v128_store( c,             c00 );
		wasm_v128_store( c + 2,         c01 );
		wasm_v128_store( c + rs_c,      c10 );
		wasm_v128_store( c + rs_c + 2,  c11 );
		wasm_v128_store( c + 2*rs_c,    c20 );
		wasm_v128_store( c + 2*rs_c+2,  c21 );
		wasm_v128_store( c + 3*rs_c,    c30 );
		wasm_v128_store( c + 3*rs_c+2,  c31 );
	}
	else
	{
		const v128_t betav = wasm_f64x2_splat( *beta );

		wasm_v128_store( c,        wasm_f64x2_add( c00, wasm_f64x2_mul( betav, wasm_v128_load( c )             ) ) );
		wasm_v128_store( c + 2,    wasm_f64x2_add( c01, wasm_f64x2_mul( betav, wasm_v128_load( c + 2 )         ) ) );
		wasm_v128_store( c + rs_c, wasm_f64x2_add( c10, wasm_f64x2_mul( betav, wasm_v128_load( c + rs_c )      ) ) );
		wasm_v128_store( c + rs_c + 2,
		               wasm_f64x2_add( c11, wasm_f64x2_mul( betav, wasm_v128_load( c + rs_c + 2 ) ) ) );
		wasm_v128_store( c + 2*rs_c,
		               wasm_f64x2_add( c20, wasm_f64x2_mul( betav, wasm_v128_load( c + 2*rs_c ) ) ) );
		wasm_v128_store( c + 2*rs_c + 2,
		               wasm_f64x2_add( c21, wasm_f64x2_mul( betav, wasm_v128_load( c + 2*rs_c + 2 ) ) ) );
		wasm_v128_store( c + 3*rs_c,
		               wasm_f64x2_add( c30, wasm_f64x2_mul( betav, wasm_v128_load( c + 3*rs_c ) ) ) );
		wasm_v128_store( c + 3*rs_c + 2,
		               wasm_f64x2_add( c31, wasm_f64x2_mul( betav, wasm_v128_load( c + 3*rs_c + 2 ) ) ) );
	}

	GEMM_UKR_FLUSH_CT( d );
}
