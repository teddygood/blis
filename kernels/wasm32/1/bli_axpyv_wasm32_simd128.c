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

// SIMD axpyv kernels for float and double, which drive GER and the
// reference fused kernels' fallbacks. As in the reference kernel, a zero
// alpha returns without reading x, and alpha == 1 adds x without the
// multiplication (the reference kernel calls addv), which gives the same
// result. Unit-stride vectors take the SIMD path; any other increments go to
// the reference kernels of this configuration, which are called by name
// rather than through the context entry that these kernels replace.
AXPYV_KER_PROT( float,  s, axpyv_wasm32_ref )
AXPYV_KER_PROT( double, d, axpyv_wasm32_ref )

void bli_saxpyv_wasm32_simd128
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	if ( bli_zero_dim1( n ) ) return;

	const float alpha = *( const float* )alpha0;

	// If alpha is zero, return.
	if ( alpha == 0.0f ) return;

	if ( incx != 1 || incy != 1 )
	{
		bli_saxpyv_wasm32_ref( conjx, n, alpha0, x0, incx, y0, incy, cntx );
		return;
	}

	// Conjugation has no effect on real inputs. x may be y itself.
	const float* x = x0;
	      float* y = y0;
	dim_t i = 0;

	if ( alpha == 1.0f )
	{
		for ( ; i + 16 <= n; i += 16 )
		{
			const v128_t y0v = wasm_f32x4_add( wasm_v128_load( y + i +  0 ), wasm_v128_load( x + i +  0 ) );
			const v128_t y1v = wasm_f32x4_add( wasm_v128_load( y + i +  4 ), wasm_v128_load( x + i +  4 ) );
			const v128_t y2v = wasm_f32x4_add( wasm_v128_load( y + i +  8 ), wasm_v128_load( x + i +  8 ) );
			const v128_t y3v = wasm_f32x4_add( wasm_v128_load( y + i + 12 ), wasm_v128_load( x + i + 12 ) );
			wasm_v128_store( y + i +  0, y0v );
			wasm_v128_store( y + i +  4, y1v );
			wasm_v128_store( y + i +  8, y2v );
			wasm_v128_store( y + i + 12, y3v );
		}
		for ( ; i + 4 <= n; i += 4 )
			wasm_v128_store( y + i, wasm_f32x4_add( wasm_v128_load( y + i ), wasm_v128_load( x + i ) ) );
		for ( ; i < n; ++i ) y[ i ] += x[ i ];
		return;
	}

	const v128_t alphav = wasm_f32x4_splat( alpha );

	for ( ; i + 16 <= n; i += 16 )
	{
		const v128_t y0v = wasm_f32x4_add( wasm_v128_load( y + i +  0 ), wasm_f32x4_mul( alphav, wasm_v128_load( x + i +  0 ) ) );
		const v128_t y1v = wasm_f32x4_add( wasm_v128_load( y + i +  4 ), wasm_f32x4_mul( alphav, wasm_v128_load( x + i +  4 ) ) );
		const v128_t y2v = wasm_f32x4_add( wasm_v128_load( y + i +  8 ), wasm_f32x4_mul( alphav, wasm_v128_load( x + i +  8 ) ) );
		const v128_t y3v = wasm_f32x4_add( wasm_v128_load( y + i + 12 ), wasm_f32x4_mul( alphav, wasm_v128_load( x + i + 12 ) ) );
		wasm_v128_store( y + i +  0, y0v );
		wasm_v128_store( y + i +  4, y1v );
		wasm_v128_store( y + i +  8, y2v );
		wasm_v128_store( y + i + 12, y3v );
	}
	for ( ; i + 4 <= n; i += 4 )
		wasm_v128_store( y + i, wasm_f32x4_add( wasm_v128_load( y + i ), wasm_f32x4_mul( alphav, wasm_v128_load( x + i ) ) ) );

	// Read only valid elements in the residual vectors.
	for ( ; i < n; ++i ) y[ i ] += alpha * x[ i ];
}

void bli_daxpyv_wasm32_simd128
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	if ( bli_zero_dim1( n ) ) return;

	const double alpha = *( const double* )alpha0;

	// If alpha is zero, return.
	if ( alpha == 0.0 ) return;

	if ( incx != 1 || incy != 1 )
	{
		bli_daxpyv_wasm32_ref( conjx, n, alpha0, x0, incx, y0, incy, cntx );
		return;
	}

	// Conjugation has no effect on real inputs. x may be y itself.
	const double* x = x0;
	      double* y = y0;
	dim_t i = 0;

	if ( alpha == 1.0 )
	{
		for ( ; i + 8 <= n; i += 8 )
		{
			const v128_t y0v = wasm_f64x2_add( wasm_v128_load( y + i + 0 ), wasm_v128_load( x + i + 0 ) );
			const v128_t y1v = wasm_f64x2_add( wasm_v128_load( y + i + 2 ), wasm_v128_load( x + i + 2 ) );
			const v128_t y2v = wasm_f64x2_add( wasm_v128_load( y + i + 4 ), wasm_v128_load( x + i + 4 ) );
			const v128_t y3v = wasm_f64x2_add( wasm_v128_load( y + i + 6 ), wasm_v128_load( x + i + 6 ) );
			wasm_v128_store( y + i + 0, y0v );
			wasm_v128_store( y + i + 2, y1v );
			wasm_v128_store( y + i + 4, y2v );
			wasm_v128_store( y + i + 6, y3v );
		}
		for ( ; i + 2 <= n; i += 2 )
			wasm_v128_store( y + i, wasm_f64x2_add( wasm_v128_load( y + i ), wasm_v128_load( x + i ) ) );
		for ( ; i < n; ++i ) y[ i ] += x[ i ];
		return;
	}

	const v128_t alphav = wasm_f64x2_splat( alpha );

	for ( ; i + 8 <= n; i += 8 )
	{
		const v128_t y0v = wasm_f64x2_add( wasm_v128_load( y + i + 0 ), wasm_f64x2_mul( alphav, wasm_v128_load( x + i + 0 ) ) );
		const v128_t y1v = wasm_f64x2_add( wasm_v128_load( y + i + 2 ), wasm_f64x2_mul( alphav, wasm_v128_load( x + i + 2 ) ) );
		const v128_t y2v = wasm_f64x2_add( wasm_v128_load( y + i + 4 ), wasm_f64x2_mul( alphav, wasm_v128_load( x + i + 4 ) ) );
		const v128_t y3v = wasm_f64x2_add( wasm_v128_load( y + i + 6 ), wasm_f64x2_mul( alphav, wasm_v128_load( x + i + 6 ) ) );
		wasm_v128_store( y + i + 0, y0v );
		wasm_v128_store( y + i + 2, y1v );
		wasm_v128_store( y + i + 4, y2v );
		wasm_v128_store( y + i + 6, y3v );
	}
	for ( ; i + 2 <= n; i += 2 )
		wasm_v128_store( y + i, wasm_f64x2_add( wasm_v128_load( y + i ), wasm_f64x2_mul( alphav, wasm_v128_load( x + i ) ) ) );

	// Read only valid elements in the residual vectors.
	for ( ; i < n; ++i ) y[ i ] += alpha * x[ i ];
}
