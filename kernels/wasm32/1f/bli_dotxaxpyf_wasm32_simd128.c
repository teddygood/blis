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

// SIMD dotxaxpyf kernels for float and double, which drive HEMV and SYMV:
//
//   y := beta * y + alpha * A^T w   (b_n elements)
//   z :=        z + alpha * A   x   (m elements)
//
// The kernels keep the summation order of the reference kernel as the
// wasm32 configuration builds it (bli_dotxaxpyf_ref.c at -O3 with
// -funsafe-math-optimizations), so HEMV/SYMV results do not change:
//
// - Only b_n == 4 with every increment equal to one is fused, as in the
//   reference; every other shape goes to the SIMD dotxf and axpyf kernels,
//   which the reference reaches through the context.
// - In the fused case the reference vectorizes its loop over the rows only
//   when m >= 8 and z overlaps neither w nor a column of A; it then keeps one
//   vector accumulator per column and adds the lanes ((l0 + l1) for double,
//   (l0 + l1) + (l2 + l3) for float) before the scalar tail. Otherwise it runs
//   the loop row by row. The kernels follow the same two paths.
// - Every element of z receives its products in increasing column order.
//
// As in the reference kernel with unit increments, y is set to zero when
// beta is zero (rather than scaled), and when m is zero or alpha is zero
// only that update of y takes place; z is then left unchanged.

#define BLIS_WASM_ALWAYS_INLINE static inline __attribute__(( always_inline ))

// For nc columns (a constant at every call): r[j] := a(:,j)^T w and
// z := z + sum_j ax[j] * a(:,j), with every element of z receiving the
// products in increasing column order.
BLIS_WASM_ALWAYS_INLINE void bli_sdotxaxpyf_wasm32_cols
     (
       const int             nc,
             dim_t           m,
       const float* restrict a, inc_t lda,
       const float* restrict w,
       const float* restrict ax,
             float* restrict z,
             float* restrict r
     )
{
	v128_t acc[ 4 ], axv[ 4 ];
	for ( int j = 0; j < nc; ++j )
	{
		acc[ j ] = wasm_f32x4_splat( 0.0f );
		axv[ j ] = wasm_f32x4_splat( ax[ j ] );
	}

	dim_t p = 0;

	for ( ; p + 4 <= m; p += 4 )
	{
		const v128_t wv = wasm_v128_load( w + p );
		      v128_t zv = wasm_v128_load( z + p );
		for ( int j = 0; j < nc; ++j )
		{
			const v128_t av = wasm_v128_load( a + p + j*lda );
			acc[ j ] = wasm_f32x4_add( acc[ j ], wasm_f32x4_mul( av, wv ) );
			zv       = wasm_f32x4_add( zv, wasm_f32x4_mul( axv[ j ], av ) );
		}
		wasm_v128_store( z + p, zv );
	}

	for ( int j = 0; j < nc; ++j )
		r[ j ] = ( wasm_f32x4_extract_lane( acc[ j ], 0 ) + wasm_f32x4_extract_lane( acc[ j ], 1 ) ) +
		         ( wasm_f32x4_extract_lane( acc[ j ], 2 ) + wasm_f32x4_extract_lane( acc[ j ], 3 ) );

	// Read only valid elements in the residual vectors.
	for ( ; p < m; ++p )
	{
		const float omega = w[ p ];
		      float zeta  = z[ p ];
		for ( int j = 0; j < nc; ++j )
		{
			const float alpha_pj = a[ p + j*lda ];
			r[ j ] += alpha_pj * omega;
			zeta   += ax[ j ] * alpha_pj;
		}
		z[ p ] = zeta;
	}
}

BLIS_WASM_ALWAYS_INLINE void bli_ddotxaxpyf_wasm32_cols
     (
       const int              nc,
             dim_t            m,
       const double* restrict a, inc_t lda,
       const double* restrict w,
       const double* restrict ax,
             double* restrict z,
             double* restrict r
     )
{
	v128_t acc[ 4 ], axv[ 4 ];
	for ( int j = 0; j < nc; ++j )
	{
		acc[ j ] = wasm_f64x2_splat( 0.0 );
		axv[ j ] = wasm_f64x2_splat( ax[ j ] );
	}

	dim_t p = 0;

	for ( ; p + 2 <= m; p += 2 )
	{
		const v128_t wv = wasm_v128_load( w + p );
		      v128_t zv = wasm_v128_load( z + p );
		for ( int j = 0; j < nc; ++j )
		{
			const v128_t av = wasm_v128_load( a + p + j*lda );
			acc[ j ] = wasm_f64x2_add( acc[ j ], wasm_f64x2_mul( av, wv ) );
			zv       = wasm_f64x2_add( zv, wasm_f64x2_mul( axv[ j ], av ) );
		}
		wasm_v128_store( z + p, zv );
	}

	for ( int j = 0; j < nc; ++j )
		r[ j ] = wasm_f64x2_extract_lane( acc[ j ], 0 ) + wasm_f64x2_extract_lane( acc[ j ], 1 );

	// Read only valid elements in the residual vectors.
	for ( ; p < m; ++p )
	{
		const double omega = w[ p ];
		      double zeta  = z[ p ];
		for ( int j = 0; j < nc; ++j )
		{
			const double alpha_pj = a[ p + j*lda ];
			r[ j ] += alpha_pj * omega;
			zeta   += ax[ j ] * alpha_pj;
		}
		z[ p ] = zeta;
	}
}

void bli_sdotxaxpyf_wasm32_simd128
     (
             conj_t  conjat,
             conj_t  conja,
             conj_t  conjw,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   w0, inc_t incw,
       const void*   x0, inc_t incx,
       const void*   beta0,
             void*   y0, inc_t incy,
             void*   z0, inc_t incz,
       const cntx_t* cntx
     )
{
	const float alpha = *( const float* )alpha0;

	if ( inca != 1 || incw != 1 || incx != 1 || incy != 1 || incz != 1 || b_n != 4 )
	{
		bli_sdotxf_wasm32_simd128
		(
		  conjat, conjw, m, b_n, alpha0, a0, inca, lda,
		  w0, incw, beta0, y0, incy, cntx
		);
		if ( bli_zero_dim1( m ) || alpha == 0.0f ) return;
		bli_saxpyf_wasm32_simd128_8
		(
		  conja, conjx, m, b_n, alpha0, a0, inca, lda,
		  x0, incx, z0, incz, cntx
		);
		return;
	}

	// Conjugation has no effect on real inputs.
	const float  beta = *( const float* )beta0;
	const float* a    = a0;
	const float* w    = w0;
	const float* x    = x0;
	      float* y    = y0;
	      float* z    = z0;

	// If beta is zero, clear y. Otherwise, scale by beta.
	if ( beta == 0.0f )
		for ( dim_t j = 0; j < 4; ++j ) y[ j ] = 0.0f;
	else
		for ( dim_t j = 0; j < 4; ++j ) y[ j ] *= beta;

	// If the vectors are empty or if alpha is zero, return early.
	if ( bli_zero_dim1( m ) || alpha == 0.0f ) return;

	float ax[ 4 ], r[ 4 ];
	for ( int k = 0; k < 4; ++k ) ax[ k ] = alpha * x[ k ];

	// The reference runs row by row for short columns, and when z overlaps w
	// or a column of A.
	const uintptr_t z_b = ( uintptr_t )z, z_e = ( uintptr_t )( z + m );
	int overlap = ( z_b < ( uintptr_t )( w + m ) && ( uintptr_t )w < z_e );
	for ( int k = 0; k < 4; ++k )
		overlap |= ( z_b < ( uintptr_t )( a + k*lda + m ) && ( uintptr_t )( a + k*lda ) < z_e );

	if ( m < 8 || overlap )
	{
		for ( int k = 0; k < 4; ++k ) r[ k ] = 0.0f;
		for ( dim_t p = 0; p < m; ++p )
		{
			for ( int k = 0; k < 4; ++k )
			{
				r[ k ] += a[ p + k*lda ] * w[ p ];
				z[ p ] += ax[ k ] * a[ p + k*lda ];
			}
		}
	}
	else
	{
		bli_sdotxaxpyf_wasm32_cols( 4, m, a, lda, w, ax, z, r );
	}

	for ( int k = 0; k < 4; ++k ) y[ k ] += alpha * r[ k ];
}

void bli_ddotxaxpyf_wasm32_simd128
     (
             conj_t  conjat,
             conj_t  conja,
             conj_t  conjw,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   w0, inc_t incw,
       const void*   x0, inc_t incx,
       const void*   beta0,
             void*   y0, inc_t incy,
             void*   z0, inc_t incz,
       const cntx_t* cntx
     )
{
	const double alpha = *( const double* )alpha0;

	if ( inca != 1 || incw != 1 || incx != 1 || incy != 1 || incz != 1 || b_n != 4 )
	{
		bli_ddotxf_wasm32_simd128
		(
		  conjat, conjw, m, b_n, alpha0, a0, inca, lda,
		  w0, incw, beta0, y0, incy, cntx
		);
		if ( bli_zero_dim1( m ) || alpha == 0.0 ) return;
		bli_daxpyf_wasm32_simd128_8
		(
		  conja, conjx, m, b_n, alpha0, a0, inca, lda,
		  x0, incx, z0, incz, cntx
		);
		return;
	}

	// Conjugation has no effect on real inputs.
	const double  beta = *( const double* )beta0;
	const double* a    = a0;
	const double* w    = w0;
	const double* x    = x0;
	      double* y    = y0;
	      double* z    = z0;

	// If beta is zero, clear y. Otherwise, scale by beta.
	if ( beta == 0.0 )
		for ( dim_t j = 0; j < 4; ++j ) y[ j ] = 0.0;
	else
		for ( dim_t j = 0; j < 4; ++j ) y[ j ] *= beta;

	// If the vectors are empty or if alpha is zero, return early.
	if ( bli_zero_dim1( m ) || alpha == 0.0 ) return;

	double ax[ 4 ], r[ 4 ];
	for ( int k = 0; k < 4; ++k ) ax[ k ] = alpha * x[ k ];

	// The reference runs row by row for short columns, and when z overlaps w
	// or a column of A.
	const uintptr_t z_b = ( uintptr_t )z, z_e = ( uintptr_t )( z + m );
	int overlap = ( z_b < ( uintptr_t )( w + m ) && ( uintptr_t )w < z_e );
	for ( int k = 0; k < 4; ++k )
		overlap |= ( z_b < ( uintptr_t )( a + k*lda + m ) && ( uintptr_t )( a + k*lda ) < z_e );

	if ( m < 8 || overlap )
	{
		for ( int k = 0; k < 4; ++k ) r[ k ] = 0.0;
		for ( dim_t p = 0; p < m; ++p )
		{
			for ( int k = 0; k < 4; ++k )
			{
				r[ k ] += a[ p + k*lda ] * w[ p ];
				z[ p ] += ax[ k ] * a[ p + k*lda ];
			}
		}
	}
	else
	{
		bli_ddotxaxpyf_wasm32_cols( 4, m, a, lda, w, ax, z, r );
	}

	for ( int k = 0; k < 4; ++k ) y[ k ] += alpha * r[ k ];
}
