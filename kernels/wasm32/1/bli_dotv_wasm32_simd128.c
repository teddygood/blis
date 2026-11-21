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

// SIMD dotv kernels for float and double. Unit-stride vectors are summed
// with four vector accumulators; any other increments go to the reference
// kernels of this configuration, which are called by name rather than
// through the context entry that these kernels replace.
DOTV_KER_PROT( float,  s, dotv_wasm32_ref )
DOTV_KER_PROT( double, d, dotv_wasm32_ref )

static inline v128_t bli_sdotv_wasm32_madd( v128_t x, v128_t y, v128_t rho )
{
	return wasm_f32x4_add( wasm_f32x4_mul( x, y ), rho );
}

static inline v128_t bli_ddotv_wasm32_madd( v128_t x, v128_t y, v128_t rho )
{
	return wasm_f64x2_add( wasm_f64x2_mul( x, y ), rho );
}

void bli_sdotv_wasm32_simd128
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   n,
       const void*   x0, inc_t incx,
       const void*   y0, inc_t incy,
             void*   rho0,
       const cntx_t* cntx
     )
{
	const float* x   = x0;
	const float* y   = y0;
	      float* rho = rho0;

	if ( bli_zero_dim1( n ) )
	{
		*rho = 0.0f;
		return;
	}

	if ( incx != 1 || incy != 1 )
	{
		bli_sdotv_wasm32_ref
		(
		  conjx, conjy, n, x0, incx, y0, incy, rho0, cntx
		);
		return;
	}

	// Conjugation has no effect on real inputs.
	const dim_t n_iter = n / 16;
	const dim_t n_left = n % 16;

	v128_t rho0v = wasm_f32x4_splat( 0.0f );
	v128_t rho1v = wasm_f32x4_splat( 0.0f );
	v128_t rho2v = wasm_f32x4_splat( 0.0f );
	v128_t rho3v = wasm_f32x4_splat( 0.0f );

	for ( dim_t i = 0; i < n_iter; ++i )
	{
		rho0v = bli_sdotv_wasm32_madd
		(
		  wasm_v128_load( x ), wasm_v128_load( y ), rho0v
		);
		rho1v = bli_sdotv_wasm32_madd
		(
		  wasm_v128_load( x + 4 ), wasm_v128_load( y + 4 ), rho1v
		);
		rho2v = bli_sdotv_wasm32_madd
		(
		  wasm_v128_load( x + 8 ), wasm_v128_load( y + 8 ), rho2v
		);
		rho3v = bli_sdotv_wasm32_madd
		(
		  wasm_v128_load( x + 12 ), wasm_v128_load( y + 12 ), rho3v
		);

		x += 16;
		y += 16;
	}

	const v128_t rho01v = wasm_f32x4_add( rho0v, rho1v );
	const v128_t rho23v = wasm_f32x4_add( rho2v, rho3v );
	const v128_t rhov   = wasm_f32x4_add( rho01v, rho23v );
	float dotxy = ( wasm_f32x4_extract_lane( rhov, 0 ) +
	                wasm_f32x4_extract_lane( rhov, 1 ) ) +
	              ( wasm_f32x4_extract_lane( rhov, 2 ) +
	                wasm_f32x4_extract_lane( rhov, 3 ) );

	// The tail reads only the remaining elements.
	for ( dim_t i = 0; i < n_left; ++i )
	{
		dotxy += x[i] * y[i];
	}

	*rho = dotxy;
}

void bli_ddotv_wasm32_simd128
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   n,
       const void*   x0, inc_t incx,
       const void*   y0, inc_t incy,
             void*   rho0,
       const cntx_t* cntx
     )
{
	const double* x   = x0;
	const double* y   = y0;
	      double* rho = rho0;

	if ( bli_zero_dim1( n ) )
	{
		*rho = 0.0;
		return;
	}

	if ( incx != 1 || incy != 1 )
	{
		bli_ddotv_wasm32_ref
		(
		  conjx, conjy, n, x0, incx, y0, incy, rho0, cntx
		);
		return;
	}

	// Conjugation has no effect on real inputs.
	const dim_t n_iter = n / 8;
	const dim_t n_left = n % 8;

	v128_t rho0v = wasm_f64x2_splat( 0.0 );
	v128_t rho1v = wasm_f64x2_splat( 0.0 );
	v128_t rho2v = wasm_f64x2_splat( 0.0 );
	v128_t rho3v = wasm_f64x2_splat( 0.0 );

	for ( dim_t i = 0; i < n_iter; ++i )
	{
		rho0v = bli_ddotv_wasm32_madd
		(
		  wasm_v128_load( x ), wasm_v128_load( y ), rho0v
		);
		rho1v = bli_ddotv_wasm32_madd
		(
		  wasm_v128_load( x + 2 ), wasm_v128_load( y + 2 ), rho1v
		);
		rho2v = bli_ddotv_wasm32_madd
		(
		  wasm_v128_load( x + 4 ), wasm_v128_load( y + 4 ), rho2v
		);
		rho3v = bli_ddotv_wasm32_madd
		(
		  wasm_v128_load( x + 6 ), wasm_v128_load( y + 6 ), rho3v
		);

		x += 8;
		y += 8;
	}

	const v128_t rho01v = wasm_f64x2_add( rho0v, rho1v );
	const v128_t rho23v = wasm_f64x2_add( rho2v, rho3v );
	const v128_t rhov   = wasm_f64x2_add( rho01v, rho23v );
	double dotxy = wasm_f64x2_extract_lane( rhov, 0 ) +
	               wasm_f64x2_extract_lane( rhov, 1 );

	// The tail reads only the remaining elements.
	for ( dim_t i = 0; i < n_left; ++i )
	{
		dotxy += x[i] * y[i];
	}

	*rho = dotxy;
}
