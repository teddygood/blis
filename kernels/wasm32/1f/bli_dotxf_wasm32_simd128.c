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

// SIMD dotxf kernels for float and double, which drive GEMV with
// transposition (and TRMV and TRSV). A call takes the SIMD path whenever
// the columns of A are contiguous (inca == 1) and incy is nonzero, for any
// increment of x and any b_n: the columns are reduced in groups of six (the
// reference fusing factor), four, two and one, and each group reads x once,
// gathering strided elements into vectors. Other calls go to the reference
// kernels of this configuration, which are called by name rather than
// through the context entry that these kernels replace.
//
// As in the reference kernel, y is set to zero when beta is zero (rather
// than scaled) and is only scaled when m is zero or alpha is zero.
DOTXF_KER_PROT( float,  s, dotxf_wasm32_ref )
DOTXF_KER_PROT( double, d, dotxf_wasm32_ref )

#define BLIS_WASM_ALWAYS_INLINE static inline __attribute__(( always_inline ))

// r[j] := a(:,j)^T x for nc columns, with nv vector accumulators per column
// and a one-vector loop for what remains. nc and nv are constants at every
// call.
BLIS_WASM_ALWAYS_INLINE void bli_sdotxf_wasm32_cols
     (
       const int             nc,
       const int             nv,
             dim_t           m,
       const float* restrict a, inc_t lda,
       const float* restrict x, inc_t incx,
             float* restrict r
     )
{
	v128_t acc[ 6 ][ 4 ];
	for ( int j = 0; j < nc; ++j )
	for ( int v = 0; v < nv; ++v ) acc[ j ][ v ] = wasm_f32x4_splat( 0.0f );

	dim_t p = 0;

	if ( incx == 1 )
	{
		for ( ; p + 4*nv <= m; p += 4*nv )
		for ( int v = 0; v < nv; ++v )
		{
			const v128_t xv = wasm_v128_load( x + p + 4*v );
			for ( int j = 0; j < nc; ++j )
				acc[ j ][ v ] = wasm_f32x4_add
				(
				  acc[ j ][ v ], wasm_f32x4_mul( wasm_v128_load( a + p + 4*v + j*lda ), xv )
				);
		}
		for ( ; nv > 1 && p + 4 <= m; p += 4 )
		{
			const v128_t xv = wasm_v128_load( x + p );
			for ( int j = 0; j < nc; ++j )
				acc[ j ][ 0 ] = wasm_f32x4_add
				(
				  acc[ j ][ 0 ], wasm_f32x4_mul( wasm_v128_load( a + p + j*lda ), xv )
				);
		}
	}
	else
	{
		for ( ; p + 4*nv <= m; p += 4*nv )
		for ( int v = 0; v < nv; ++v )
		{
			const float* restrict xp = x + ( p + 4*v )*incx;
			const v128_t xv = wasm_f32x4_make( xp[ 0 ], xp[ incx ], xp[ 2*incx ], xp[ 3*incx ] );
			for ( int j = 0; j < nc; ++j )
				acc[ j ][ v ] = wasm_f32x4_add
				(
				  acc[ j ][ v ], wasm_f32x4_mul( wasm_v128_load( a + p + 4*v + j*lda ), xv )
				);
		}
		for ( ; nv > 1 && p + 4 <= m; p += 4 )
		{
			const v128_t xv = wasm_f32x4_make( x[ p*incx ], x[ ( p + 1 )*incx ],
			                                   x[ ( p + 2 )*incx ], x[ ( p + 3 )*incx ] );
			for ( int j = 0; j < nc; ++j )
				acc[ j ][ 0 ] = wasm_f32x4_add
				(
				  acc[ j ][ 0 ], wasm_f32x4_mul( wasm_v128_load( a + p + j*lda ), xv )
				);
		}
	}

	for ( int j = 0; j < nc; ++j )
	{
		v128_t s = acc[ j ][ 0 ];
		for ( int v = 1; v < nv; ++v ) s = wasm_f32x4_add( s, acc[ j ][ v ] );
		r[ j ] = ( wasm_f32x4_extract_lane( s, 0 ) + wasm_f32x4_extract_lane( s, 1 ) ) +
		         ( wasm_f32x4_extract_lane( s, 2 ) + wasm_f32x4_extract_lane( s, 3 ) );
	}

	// Read only valid elements in the residual vector.
	for ( ; p < m; ++p )
	{
		const float chi = x[ p*incx ];
		for ( int j = 0; j < nc; ++j ) r[ j ] += a[ p + j*lda ] * chi;
	}
}

BLIS_WASM_ALWAYS_INLINE void bli_ddotxf_wasm32_cols
     (
       const int              nc,
       const int              nv,
             dim_t            m,
       const double* restrict a, inc_t lda,
       const double* restrict x, inc_t incx,
             double* restrict r
     )
{
	v128_t acc[ 6 ][ 4 ];
	for ( int j = 0; j < nc; ++j )
	for ( int v = 0; v < nv; ++v ) acc[ j ][ v ] = wasm_f64x2_splat( 0.0 );

	dim_t p = 0;

	if ( incx == 1 )
	{
		for ( ; p + 2*nv <= m; p += 2*nv )
		for ( int v = 0; v < nv; ++v )
		{
			const v128_t xv = wasm_v128_load( x + p + 2*v );
			for ( int j = 0; j < nc; ++j )
				acc[ j ][ v ] = wasm_f64x2_add
				(
				  acc[ j ][ v ], wasm_f64x2_mul( wasm_v128_load( a + p + 2*v + j*lda ), xv )
				);
		}
		for ( ; nv > 1 && p + 2 <= m; p += 2 )
		{
			const v128_t xv = wasm_v128_load( x + p );
			for ( int j = 0; j < nc; ++j )
				acc[ j ][ 0 ] = wasm_f64x2_add
				(
				  acc[ j ][ 0 ], wasm_f64x2_mul( wasm_v128_load( a + p + j*lda ), xv )
				);
		}
	}
	else
	{
		for ( ; p + 2*nv <= m; p += 2*nv )
		for ( int v = 0; v < nv; ++v )
		{
			const double* restrict xp = x + ( p + 2*v )*incx;
			const v128_t xv = wasm_f64x2_make( xp[ 0 ], xp[ incx ] );
			for ( int j = 0; j < nc; ++j )
				acc[ j ][ v ] = wasm_f64x2_add
				(
				  acc[ j ][ v ], wasm_f64x2_mul( wasm_v128_load( a + p + 2*v + j*lda ), xv )
				);
		}
		for ( ; nv > 1 && p + 2 <= m; p += 2 )
		{
			const v128_t xv = wasm_f64x2_make( x[ p*incx ], x[ ( p + 1 )*incx ] );
			for ( int j = 0; j < nc; ++j )
				acc[ j ][ 0 ] = wasm_f64x2_add
				(
				  acc[ j ][ 0 ], wasm_f64x2_mul( wasm_v128_load( a + p + j*lda ), xv )
				);
		}
	}

	for ( int j = 0; j < nc; ++j )
	{
		v128_t s = acc[ j ][ 0 ];
		for ( int v = 1; v < nv; ++v ) s = wasm_f64x2_add( s, acc[ j ][ v ] );
		r[ j ] = wasm_f64x2_extract_lane( s, 0 ) + wasm_f64x2_extract_lane( s, 1 );
	}

	// Read only valid elements in the residual vector.
	for ( ; p < m; ++p )
	{
		const double chi = x[ p*incx ];
		for ( int j = 0; j < nc; ++j ) r[ j ] += a[ p + j*lda ] * chi;
	}
}

void bli_sdotxf_wasm32_simd128
     (
             conj_t  conjat,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
       const void*   beta0,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	if ( inca != 1 || incy == 0 )
	{
		bli_sdotxf_wasm32_ref
		(
		  conjat, conjx, m, b_n, alpha0, a0, inca, lda,
		  x0, incx, beta0, y0, incy, cntx
		);
		return;
	}

	// Conjugation has no effect on real inputs.
	const float           alpha = *( const float* )alpha0;
	const float           beta  = *( const float* )beta0;
	const float* restrict a     = a0;
	const float* restrict x     = x0;
	      float* restrict y     = y0;

	// If beta is zero, clear y. Otherwise, scale by beta.
	if ( beta == 0.0f )
		for ( dim_t j = 0; j < b_n; ++j ) y[ j*incy ] = 0.0f;
	else
		for ( dim_t j = 0; j < b_n; ++j ) y[ j*incy ] *= beta;

	// If the vectors are empty or if alpha is zero, return early.
	if ( bli_zero_dim1( m ) || alpha == 0.0f ) return;

	float r[ 6 ];
	dim_t j = 0;

	for ( ; j + 6 <= b_n; j += 6 )
	{
		bli_sdotxf_wasm32_cols( 6, 1, m, a + j*lda, lda, x, incx, r );
		for ( int k = 0; k < 6; ++k ) y[ ( j + k )*incy ] += alpha * r[ k ];
	}
	if ( j + 4 <= b_n )
	{
		bli_sdotxf_wasm32_cols( 4, 2, m, a + j*lda, lda, x, incx, r );
		for ( int k = 0; k < 4; ++k ) y[ ( j + k )*incy ] += alpha * r[ k ];
		j += 4;
	}
	if ( j + 2 <= b_n )
	{
		bli_sdotxf_wasm32_cols( 2, 2, m, a + j*lda, lda, x, incx, r );
		for ( int k = 0; k < 2; ++k ) y[ ( j + k )*incy ] += alpha * r[ k ];
		j += 2;
	}
	if ( j < b_n )
	{
		bli_sdotxf_wasm32_cols( 1, 4, m, a + j*lda, lda, x, incx, r );
		y[ j*incy ] += alpha * r[ 0 ];
	}
}

void bli_ddotxf_wasm32_simd128
     (
             conj_t  conjat,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
       const void*   beta0,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	if ( inca != 1 || incy == 0 )
	{
		bli_ddotxf_wasm32_ref
		(
		  conjat, conjx, m, b_n, alpha0, a0, inca, lda,
		  x0, incx, beta0, y0, incy, cntx
		);
		return;
	}

	// Conjugation has no effect on real inputs.
	const double           alpha = *( const double* )alpha0;
	const double           beta  = *( const double* )beta0;
	const double* restrict a     = a0;
	const double* restrict x     = x0;
	      double* restrict y     = y0;

	// If beta is zero, clear y. Otherwise, scale by beta.
	if ( beta == 0.0 )
		for ( dim_t j = 0; j < b_n; ++j ) y[ j*incy ] = 0.0;
	else
		for ( dim_t j = 0; j < b_n; ++j ) y[ j*incy ] *= beta;

	// If the vectors are empty or if alpha is zero, return early.
	if ( bli_zero_dim1( m ) || alpha == 0.0 ) return;

	double r[ 6 ];
	dim_t  j = 0;

	for ( ; j + 6 <= b_n; j += 6 )
	{
		bli_ddotxf_wasm32_cols( 6, 1, m, a + j*lda, lda, x, incx, r );
		for ( int k = 0; k < 6; ++k ) y[ ( j + k )*incy ] += alpha * r[ k ];
	}
	if ( j + 4 <= b_n )
	{
		bli_ddotxf_wasm32_cols( 4, 2, m, a + j*lda, lda, x, incx, r );
		for ( int k = 0; k < 4; ++k ) y[ ( j + k )*incy ] += alpha * r[ k ];
		j += 4;
	}
	if ( j + 2 <= b_n )
	{
		bli_ddotxf_wasm32_cols( 2, 2, m, a + j*lda, lda, x, incx, r );
		for ( int k = 0; k < 2; ++k ) y[ ( j + k )*incy ] += alpha * r[ k ];
		j += 2;
	}
	if ( j < b_n )
	{
		bli_ddotxf_wasm32_cols( 1, 4, m, a + j*lda, lda, x, incx, r );
		y[ j*incy ] += alpha * r[ 0 ];
	}
}
