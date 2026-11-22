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

// dotxv kernels for float and double built on the SIMD dotv kernels of this
// configuration, which take unit-stride vectors and pass any other
// increments to the reference dotv kernels:
//
//   rho := beta * rho + alpha * x^T y
//
// As in the reference kernel, rho is set to zero when beta is zero (rather
// than scaled), and x and y are not read when n is zero or alpha is zero.

void bli_sdotxv_wasm32_simd128
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
       const void*   y0, inc_t incy,
       const void*   beta0,
             void*   rho0,
       const cntx_t* cntx
     )
{
	const float alpha = *( const float* )alpha0;
	const float beta  = *( const float* )beta0;
	      float* rho  = rho0;

	// If beta is zero, clear rho. Otherwise, scale by beta.
	if ( beta == 0.0f ) *rho = 0.0f;
	else                *rho *= beta;

	// If the vectors are empty or if alpha is zero, return early.
	if ( bli_zero_dim1( n ) || alpha == 0.0f ) return;

	// Conjugation has no effect on real inputs.
	float dotxy;
	bli_sdotv_wasm32_simd128( conjx, conjy, n, x0, incx, y0, incy, &dotxy, cntx );

	*rho += alpha * dotxy;
}

void bli_ddotxv_wasm32_simd128
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
       const void*   y0, inc_t incy,
       const void*   beta0,
             void*   rho0,
       const cntx_t* cntx
     )
{
	const double alpha = *( const double* )alpha0;
	const double beta  = *( const double* )beta0;
	      double* rho  = rho0;

	// If beta is zero, clear rho. Otherwise, scale by beta.
	if ( beta == 0.0 ) *rho = 0.0;
	else               *rho *= beta;

	// If the vectors are empty or if alpha is zero, return early.
	if ( bli_zero_dim1( n ) || alpha == 0.0 ) return;

	// Conjugation has no effect on real inputs.
	double dotxy;
	bli_ddotv_wasm32_simd128( conjx, conjy, n, x0, incx, y0, incy, &dotxy, cntx );

	*rho += alpha * dotxy;
}
