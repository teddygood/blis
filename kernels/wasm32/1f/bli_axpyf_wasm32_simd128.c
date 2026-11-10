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
// of 8, which drives column-major GEMV without transposition. Calls with
// unit increments and b_n == 8 take the SIMD path; all others go to the
// reference kernels of this configuration, which are called by name rather
// than through the context entry that these kernels replace.
AXPYF_KER_PROT( float,  s, axpyf_wasm32_ref )
AXPYF_KER_PROT( double, d, axpyf_wasm32_ref )

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

	if ( inca != 1 || incx != 1 || incy != 1 || b_n != 8 )
	{
		bli_saxpyf_wasm32_ref
		(
		  conja, conjx, m, b_n, alpha0, a0, inca, lda,
		  x0, incx, y0, incy, cntx
		);
		return;
	}

	const float* restrict alpha = alpha0;
	const float* restrict a     = a0;
	const float* restrict x     = x0;
	      float* restrict y     = y0;

	// Use four-column passes only for the longer output vectors.
	// AF remains eight and each element keeps the same accumulation order.
	if ( m >= 1024 )
	{
		for ( dim_t j = 0; j < 8; j += 4 )
		{
			const float* ap = a + j * lda;
			const float ax0 = *alpha * x[j + 0];
			const float ax1 = *alpha * x[j + 1];
			const float ax2 = *alpha * x[j + 2];
			const float ax3 = *alpha * x[j + 3];
			const v128_t ax0v = wasm_f32x4_splat( ax0 );
			const v128_t ax1v = wasm_f32x4_splat( ax1 );
			const v128_t ax2v = wasm_f32x4_splat( ax2 );
			const v128_t ax3v = wasm_f32x4_splat( ax3 );
			dim_t i = 0;
			const dim_t m_main = m / 16 * 16;
			for ( ; i < m_main; i += 16 )
			{
				{
					v128_t yv = wasm_v128_load( y + i + 0 );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax0v, wasm_v128_load( ap + i + 0 + 0*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax1v, wasm_v128_load( ap + i + 0 + 1*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax2v, wasm_v128_load( ap + i + 0 + 2*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax3v, wasm_v128_load( ap + i + 0 + 3*lda ) ) );
					wasm_v128_store( y + i + 0, yv );
				}
				{
					v128_t yv = wasm_v128_load( y + i + 4 );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax0v, wasm_v128_load( ap + i + 4 + 0*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax1v, wasm_v128_load( ap + i + 4 + 1*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax2v, wasm_v128_load( ap + i + 4 + 2*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax3v, wasm_v128_load( ap + i + 4 + 3*lda ) ) );
					wasm_v128_store( y + i + 4, yv );
				}
				{
					v128_t yv = wasm_v128_load( y + i + 8 );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax0v, wasm_v128_load( ap + i + 8 + 0*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax1v, wasm_v128_load( ap + i + 8 + 1*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax2v, wasm_v128_load( ap + i + 8 + 2*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax3v, wasm_v128_load( ap + i + 8 + 3*lda ) ) );
					wasm_v128_store( y + i + 8, yv );
				}
				{
					v128_t yv = wasm_v128_load( y + i + 12 );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax0v, wasm_v128_load( ap + i + 12 + 0*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax1v, wasm_v128_load( ap + i + 12 + 1*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax2v, wasm_v128_load( ap + i + 12 + 2*lda ) ) );
					yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax3v, wasm_v128_load( ap + i + 12 + 3*lda ) ) );
					wasm_v128_store( y + i + 12, yv );
				}
			}
			const dim_t m_vector = m / 4 * 4;
			for ( ; i < m_vector; i += 4 )
			{
				v128_t yv = wasm_v128_load( y + i );
				yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax0v, wasm_v128_load( ap + i + 0*lda ) ) );
				yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax1v, wasm_v128_load( ap + i + 1*lda ) ) );
				yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax2v, wasm_v128_load( ap + i + 2*lda ) ) );
				yv = wasm_f32x4_add( yv, wasm_f32x4_mul( ax3v, wasm_v128_load( ap + i + 3*lda ) ) );
				wasm_v128_store( y + i, yv );
			}
			for ( ; i < m; ++i )
			{
				float chi = y[i];
				chi += ax0 * ap[i + 0*lda];
				chi += ax1 * ap[i + 1*lda];
				chi += ax2 * ap[i + 2*lda];
				chi += ax3 * ap[i + 3*lda];
				y[i] = chi;
			}
		}
		return;
	}

	// Conjugation has no effect on real inputs. Keep the eight-column
	// accumulation order and defer the output store until it is complete.
	const float ax0 = *alpha * x[0];
	const float ax1 = *alpha * x[1];
	const float ax2 = *alpha * x[2];
	const float ax3 = *alpha * x[3];
	const float ax4 = *alpha * x[4];
	const float ax5 = *alpha * x[5];
	const float ax6 = *alpha * x[6];
	const float ax7 = *alpha * x[7];

	const v128_t ax0v = wasm_f32x4_splat( ax0 );
	const v128_t ax1v = wasm_f32x4_splat( ax1 );
	const v128_t ax2v = wasm_f32x4_splat( ax2 );
	const v128_t ax3v = wasm_f32x4_splat( ax3 );
	const v128_t ax4v = wasm_f32x4_splat( ax4 );
	const v128_t ax5v = wasm_f32x4_splat( ax5 );
	const v128_t ax6v = wasm_f32x4_splat( ax6 );
	const v128_t ax7v = wasm_f32x4_splat( ax7 );

	const dim_t m_iter = m / 4 * 4;
	dim_t i = 0;

	for ( ; i < m_iter; i += 4 )
	{
		v128_t yv = wasm_v128_load( y + i );
		yv = wasm_f32x4_add
		(
		  yv, wasm_f32x4_mul( ax0v, wasm_v128_load( a + i + 0*lda ) )
		);
		yv = wasm_f32x4_add
		(
		  yv, wasm_f32x4_mul( ax1v, wasm_v128_load( a + i + 1*lda ) )
		);
		yv = wasm_f32x4_add
		(
		  yv, wasm_f32x4_mul( ax2v, wasm_v128_load( a + i + 2*lda ) )
		);
		yv = wasm_f32x4_add
		(
		  yv, wasm_f32x4_mul( ax3v, wasm_v128_load( a + i + 3*lda ) )
		);
		yv = wasm_f32x4_add
		(
		  yv, wasm_f32x4_mul( ax4v, wasm_v128_load( a + i + 4*lda ) )
		);
		yv = wasm_f32x4_add
		(
		  yv, wasm_f32x4_mul( ax5v, wasm_v128_load( a + i + 5*lda ) )
		);
		yv = wasm_f32x4_add
		(
		  yv, wasm_f32x4_mul( ax6v, wasm_v128_load( a + i + 6*lda ) )
		);
		yv = wasm_f32x4_add
		(
		  yv, wasm_f32x4_mul( ax7v, wasm_v128_load( a + i + 7*lda ) )
		);

		wasm_v128_store( y + i, yv );
	}

	// Read only valid elements in the residual vector.
	for ( ; i < m; ++i )
	{
		float chi = y[i];

		chi += ax0 * a[i + 0*lda];
		chi += ax1 * a[i + 1*lda];
		chi += ax2 * a[i + 2*lda];
		chi += ax3 * a[i + 3*lda];
		chi += ax4 * a[i + 4*lda];
		chi += ax5 * a[i + 5*lda];
		chi += ax6 * a[i + 6*lda];
		chi += ax7 * a[i + 7*lda];

		y[i] = chi;
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

	if ( inca != 1 || incx != 1 || incy != 1 || b_n != 8 )
	{
		bli_daxpyf_wasm32_ref
		(
		  conja, conjx, m, b_n, alpha0, a0, inca, lda,
		  x0, incx, y0, incy, cntx
		);
		return;
	}

	const double* restrict alpha = alpha0;
	const double* restrict a     = a0;
	const double* restrict x     = x0;
	      double* restrict y     = y0;

	// Use four-column passes only for the longer output vectors.
	// AF remains eight and each element keeps the same accumulation order.
	if ( m >= 512 )
	{
		for ( dim_t j = 0; j < 8; j += 4 )
		{
			const double* ap = a + j * lda;
			const double ax0 = *alpha * x[j + 0];
			const double ax1 = *alpha * x[j + 1];
			const double ax2 = *alpha * x[j + 2];
			const double ax3 = *alpha * x[j + 3];
			const v128_t ax0v = wasm_f64x2_splat( ax0 );
			const v128_t ax1v = wasm_f64x2_splat( ax1 );
			const v128_t ax2v = wasm_f64x2_splat( ax2 );
			const v128_t ax3v = wasm_f64x2_splat( ax3 );
			dim_t i = 0;
			const dim_t m_main = m / 8 * 8;
			for ( ; i < m_main; i += 8 )
			{
				{
					v128_t yv = wasm_v128_load( y + i + 0 );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax0v, wasm_v128_load( ap + i + 0 + 0*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax1v, wasm_v128_load( ap + i + 0 + 1*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax2v, wasm_v128_load( ap + i + 0 + 2*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax3v, wasm_v128_load( ap + i + 0 + 3*lda ) ) );
					wasm_v128_store( y + i + 0, yv );
				}
				{
					v128_t yv = wasm_v128_load( y + i + 2 );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax0v, wasm_v128_load( ap + i + 2 + 0*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax1v, wasm_v128_load( ap + i + 2 + 1*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax2v, wasm_v128_load( ap + i + 2 + 2*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax3v, wasm_v128_load( ap + i + 2 + 3*lda ) ) );
					wasm_v128_store( y + i + 2, yv );
				}
				{
					v128_t yv = wasm_v128_load( y + i + 4 );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax0v, wasm_v128_load( ap + i + 4 + 0*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax1v, wasm_v128_load( ap + i + 4 + 1*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax2v, wasm_v128_load( ap + i + 4 + 2*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax3v, wasm_v128_load( ap + i + 4 + 3*lda ) ) );
					wasm_v128_store( y + i + 4, yv );
				}
				{
					v128_t yv = wasm_v128_load( y + i + 6 );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax0v, wasm_v128_load( ap + i + 6 + 0*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax1v, wasm_v128_load( ap + i + 6 + 1*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax2v, wasm_v128_load( ap + i + 6 + 2*lda ) ) );
					yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax3v, wasm_v128_load( ap + i + 6 + 3*lda ) ) );
					wasm_v128_store( y + i + 6, yv );
				}
			}
			const dim_t m_vector = m / 2 * 2;
			for ( ; i < m_vector; i += 2 )
			{
				v128_t yv = wasm_v128_load( y + i );
				yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax0v, wasm_v128_load( ap + i + 0*lda ) ) );
				yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax1v, wasm_v128_load( ap + i + 1*lda ) ) );
				yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax2v, wasm_v128_load( ap + i + 2*lda ) ) );
				yv = wasm_f64x2_add( yv, wasm_f64x2_mul( ax3v, wasm_v128_load( ap + i + 3*lda ) ) );
				wasm_v128_store( y + i, yv );
			}
			for ( ; i < m; ++i )
			{
				double chi = y[i];
				chi += ax0 * ap[i + 0*lda];
				chi += ax1 * ap[i + 1*lda];
				chi += ax2 * ap[i + 2*lda];
				chi += ax3 * ap[i + 3*lda];
				y[i] = chi;
			}
		}
		return;
	}

	// Conjugation has no effect on real inputs. Keep the eight-column
	// accumulation order and defer the output store until it is complete.
	const double ax0 = *alpha * x[0];
	const double ax1 = *alpha * x[1];
	const double ax2 = *alpha * x[2];
	const double ax3 = *alpha * x[3];
	const double ax4 = *alpha * x[4];
	const double ax5 = *alpha * x[5];
	const double ax6 = *alpha * x[6];
	const double ax7 = *alpha * x[7];

	const v128_t ax0v = wasm_f64x2_splat( ax0 );
	const v128_t ax1v = wasm_f64x2_splat( ax1 );
	const v128_t ax2v = wasm_f64x2_splat( ax2 );
	const v128_t ax3v = wasm_f64x2_splat( ax3 );
	const v128_t ax4v = wasm_f64x2_splat( ax4 );
	const v128_t ax5v = wasm_f64x2_splat( ax5 );
	const v128_t ax6v = wasm_f64x2_splat( ax6 );
	const v128_t ax7v = wasm_f64x2_splat( ax7 );

	const dim_t m_iter = m / 2 * 2;
	dim_t i = 0;

	for ( ; i < m_iter; i += 2 )
	{
		v128_t yv = wasm_v128_load( y + i );
		yv = wasm_f64x2_add
		(
		  yv, wasm_f64x2_mul( ax0v, wasm_v128_load( a + i + 0*lda ) )
		);
		yv = wasm_f64x2_add
		(
		  yv, wasm_f64x2_mul( ax1v, wasm_v128_load( a + i + 1*lda ) )
		);
		yv = wasm_f64x2_add
		(
		  yv, wasm_f64x2_mul( ax2v, wasm_v128_load( a + i + 2*lda ) )
		);
		yv = wasm_f64x2_add
		(
		  yv, wasm_f64x2_mul( ax3v, wasm_v128_load( a + i + 3*lda ) )
		);
		yv = wasm_f64x2_add
		(
		  yv, wasm_f64x2_mul( ax4v, wasm_v128_load( a + i + 4*lda ) )
		);
		yv = wasm_f64x2_add
		(
		  yv, wasm_f64x2_mul( ax5v, wasm_v128_load( a + i + 5*lda ) )
		);
		yv = wasm_f64x2_add
		(
		  yv, wasm_f64x2_mul( ax6v, wasm_v128_load( a + i + 6*lda ) )
		);
		yv = wasm_f64x2_add
		(
		  yv, wasm_f64x2_mul( ax7v, wasm_v128_load( a + i + 7*lda ) )
		);

		wasm_v128_store( y + i, yv );
	}

	// Read only valid elements in the residual vector.
	for ( ; i < m; ++i )
	{
		double chi = y[i];

		chi += ax0 * a[i + 0*lda];
		chi += ax1 * a[i + 1*lda];
		chi += ax2 * a[i + 2*lda];
		chi += ax3 * a[i + 3*lda];
		chi += ax4 * a[i + 4*lda];
		chi += ax5 * a[i + 5*lda];
		chi += ax6 * a[i + 6*lda];
		chi += ax7 * a[i + 7*lda];

		y[i] = chi;
	}
}
