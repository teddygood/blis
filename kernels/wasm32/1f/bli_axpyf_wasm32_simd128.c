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

// SIMD axpyf kernels for float and double with the reference fusing factor
// of 8, which drives GEMV without transposition (and TRMV and TRSV). A call
// takes the SIMD path whenever the columns of A are contiguous (inca == 1)
// and incy is nonzero, for any increment of x and any b_n: the columns are
// applied in passes of eight, four, two and one, and each pass reads and
// writes y once. Strided elements of y are gathered into vectors and
// scattered back after the pass. Other calls go to the reference kernels of
// this configuration, which are called by name rather than through the
// context entry that these kernels replace.
//
// Every element of y receives the products of the columns in increasing
// column order, like the reference kernel with unit increments. Unlike the
// reference kernel's fallback for other shapes, which calls axpyv once per
// column and skips a column whose alpha * chi is zero, the SIMD path skips
// no column, so a NaN or infinity in A or x reaches y whatever the
// increments of x and y. (Calls with inca != 1 or incy == 0 still take the
// reference kernel and its fallback.)
AXPYF_KER_PROT( float,  s, axpyf_wasm32_ref )
AXPYF_KER_PROT( double, d, axpyf_wasm32_ref )

#define BLIS_WASM_ALWAYS_INLINE static inline __attribute__(( always_inline ))

// y := y + sum_j ax[j] * a(:,j) for nc columns, with nv vectors per
// iteration of the unit-stride loop. nc and nv are constants at every call.
BLIS_WASM_ALWAYS_INLINE void bli_saxpyf_wasm32_cols
     (
       const int             nc,
       const int             nv,
             dim_t           m,
       const float* restrict ax,
       const float* restrict a, inc_t lda,
             float* restrict y, inc_t incy
     )
{
	v128_t axv[ 8 ];
	for ( int j = 0; j < nc; ++j ) axv[ j ] = wasm_f32x4_splat( ax[ j ] );

	dim_t i = 0;

	if ( incy == 1 )
	{
		for ( ; i + 4*nv <= m; i += 4*nv )
		for ( int v = 0; v < nv; ++v )
		{
			v128_t yv = wasm_v128_load( y + i + 4*v );
			for ( int j = 0; j < nc; ++j )
				yv = wasm_f32x4_add
				(
				  yv, wasm_f32x4_mul( axv[ j ], wasm_v128_load( a + i + 4*v + j*lda ) )
				);
			wasm_v128_store( y + i + 4*v, yv );
		}
		for ( ; i + 4 <= m; i += 4 )
		{
			v128_t yv = wasm_v128_load( y + i );
			for ( int j = 0; j < nc; ++j )
				yv = wasm_f32x4_add
				(
				  yv, wasm_f32x4_mul( axv[ j ], wasm_v128_load( a + i + j*lda ) )
				);
			wasm_v128_store( y + i, yv );
		}
	}
	else
	{
		for ( ; i + 4 <= m; i += 4 )
		{
			float* y0 = y + ( i + 0 )*incy;
			float* y1 = y + ( i + 1 )*incy;
			float* y2 = y + ( i + 2 )*incy;
			float* y3 = y + ( i + 3 )*incy;
			v128_t yv = wasm_f32x4_make( *y0, *y1, *y2, *y3 );
			for ( int j = 0; j < nc; ++j )
				yv = wasm_f32x4_add
				(
				  yv, wasm_f32x4_mul( axv[ j ], wasm_v128_load( a + i + j*lda ) )
				);
			*y0 = wasm_f32x4_extract_lane( yv, 0 );
			*y1 = wasm_f32x4_extract_lane( yv, 1 );
			*y2 = wasm_f32x4_extract_lane( yv, 2 );
			*y3 = wasm_f32x4_extract_lane( yv, 3 );
		}
	}

	// Read only valid elements in the residual vector.
	for ( ; i < m; ++i )
	{
		float chi = y[ i*incy ];
		for ( int j = 0; j < nc; ++j ) chi += ax[ j ] * a[ i + j*lda ];
		y[ i*incy ] = chi;
	}
}

BLIS_WASM_ALWAYS_INLINE void bli_daxpyf_wasm32_cols
     (
       const int              nc,
       const int              nv,
             dim_t            m,
       const double* restrict ax,
       const double* restrict a, inc_t lda,
             double* restrict y, inc_t incy
     )
{
	v128_t axv[ 8 ];
	for ( int j = 0; j < nc; ++j ) axv[ j ] = wasm_f64x2_splat( ax[ j ] );

	dim_t i = 0;

	if ( incy == 1 )
	{
		for ( ; i + 2*nv <= m; i += 2*nv )
		for ( int v = 0; v < nv; ++v )
		{
			v128_t yv = wasm_v128_load( y + i + 2*v );
			for ( int j = 0; j < nc; ++j )
				yv = wasm_f64x2_add
				(
				  yv, wasm_f64x2_mul( axv[ j ], wasm_v128_load( a + i + 2*v + j*lda ) )
				);
			wasm_v128_store( y + i + 2*v, yv );
		}
		for ( ; i + 2 <= m; i += 2 )
		{
			v128_t yv = wasm_v128_load( y + i );
			for ( int j = 0; j < nc; ++j )
				yv = wasm_f64x2_add
				(
				  yv, wasm_f64x2_mul( axv[ j ], wasm_v128_load( a + i + j*lda ) )
				);
			wasm_v128_store( y + i, yv );
		}
	}
	else
	{
		for ( ; i + 2 <= m; i += 2 )
		{
			double* y0 = y + ( i + 0 )*incy;
			double* y1 = y + ( i + 1 )*incy;
			v128_t yv = wasm_f64x2_make( *y0, *y1 );
			for ( int j = 0; j < nc; ++j )
				yv = wasm_f64x2_add
				(
				  yv, wasm_f64x2_mul( axv[ j ], wasm_v128_load( a + i + j*lda ) )
				);
			*y0 = wasm_f64x2_extract_lane( yv, 0 );
			*y1 = wasm_f64x2_extract_lane( yv, 1 );
		}
	}

	// Read only valid elements in the residual vector.
	for ( ; i < m; ++i )
	{
		double chi = y[ i*incy ];
		for ( int j = 0; j < nc; ++j ) chi += ax[ j ] * a[ i + j*lda ];
		y[ i*incy ] = chi;
	}
}

void bli_saxpyf_wasm32_simd128_8
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	if ( bli_zero_dim1( m ) ) return;

	if ( inca != 1 || incy == 0 )
	{
		bli_saxpyf_wasm32_ref
		(
		  conja, conjx, m, b_n, alpha0, a0, inca, lda,
		  x0, incx, y0, incy, cntx
		);
		return;
	}

	// Conjugation has no effect on real inputs.
	const float           alpha = *( const float* )alpha0;
	const float* restrict a     = a0;
	const float* restrict x     = x0;
	      float* restrict y     = y0;

	float ax[ 8 ];
	dim_t j = 0;

	for ( ; j + 8 <= b_n; j += 8 )
	{
		for ( int k = 0; k < 8; ++k ) ax[ k ] = alpha * x[ ( j + k )*incx ];

		// Long unit-stride vectors take two four-column passes, which
		// keep fewer streams in flight; the summation order is the same.
		if ( m >= 1024 && incy == 1 )
		{
			bli_saxpyf_wasm32_cols( 4, 4, m, ax + 0, a + ( j + 0 )*lda, lda, y, incy );
			bli_saxpyf_wasm32_cols( 4, 4, m, ax + 4, a + ( j + 4 )*lda, lda, y, incy );
		}
		else
		{
			bli_saxpyf_wasm32_cols( 8, 1, m, ax, a + j*lda, lda, y, incy );
		}
	}
	if ( j + 4 <= b_n )
	{
		for ( int k = 0; k < 4; ++k ) ax[ k ] = alpha * x[ ( j + k )*incx ];
		bli_saxpyf_wasm32_cols( 4, 4, m, ax, a + j*lda, lda, y, incy );
		j += 4;
	}
	if ( j + 2 <= b_n )
	{
		for ( int k = 0; k < 2; ++k ) ax[ k ] = alpha * x[ ( j + k )*incx ];
		bli_saxpyf_wasm32_cols( 2, 4, m, ax, a + j*lda, lda, y, incy );
		j += 2;
	}
	if ( j < b_n )
	{
		ax[ 0 ] = alpha * x[ j*incx ];
		bli_saxpyf_wasm32_cols( 1, 4, m, ax, a + j*lda, lda, y, incy );
	}
}

void bli_daxpyf_wasm32_simd128_8
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	if ( bli_zero_dim1( m ) ) return;

	if ( inca != 1 || incy == 0 )
	{
		bli_daxpyf_wasm32_ref
		(
		  conja, conjx, m, b_n, alpha0, a0, inca, lda,
		  x0, incx, y0, incy, cntx
		);
		return;
	}

	// Conjugation has no effect on real inputs.
	const double           alpha = *( const double* )alpha0;
	const double* restrict a     = a0;
	const double* restrict x     = x0;
	      double* restrict y     = y0;

	double ax[ 8 ];
	dim_t  j = 0;

	for ( ; j + 8 <= b_n; j += 8 )
	{
		for ( int k = 0; k < 8; ++k ) ax[ k ] = alpha * x[ ( j + k )*incx ];

		// Long unit-stride vectors take two four-column passes, which
		// keep fewer streams in flight; the summation order is the same.
		if ( m >= 512 && incy == 1 )
		{
			bli_daxpyf_wasm32_cols( 4, 4, m, ax + 0, a + ( j + 0 )*lda, lda, y, incy );
			bli_daxpyf_wasm32_cols( 4, 4, m, ax + 4, a + ( j + 4 )*lda, lda, y, incy );
		}
		else
		{
			bli_daxpyf_wasm32_cols( 8, 1, m, ax, a + j*lda, lda, y, incy );
		}
	}
	if ( j + 4 <= b_n )
	{
		for ( int k = 0; k < 4; ++k ) ax[ k ] = alpha * x[ ( j + k )*incx ];
		bli_daxpyf_wasm32_cols( 4, 4, m, ax, a + j*lda, lda, y, incy );
		j += 4;
	}
	if ( j + 2 <= b_n )
	{
		for ( int k = 0; k < 2; ++k ) ax[ k ] = alpha * x[ ( j + k )*incx ];
		bli_daxpyf_wasm32_cols( 2, 4, m, ax, a + j*lda, lda, y, incy );
		j += 2;
	}
	if ( j < b_n )
	{
		ax[ 0 ] = alpha * x[ j*incx ];
		bli_daxpyf_wasm32_cols( 1, 4, m, ax, a + j*lda, lda, y, incy );
	}
}
