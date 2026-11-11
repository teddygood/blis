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

// trsm microkernels for the 4x4 double tile of bli_dgemm_wasm32_simd128_4x4.
// The reference gemmtrsm microkernel updates B11 with that gemm kernel and
// then calls one of these to compute
//
//   B11 := inv( A11 ) * B11,  C11 := B11
//
// on a full 4x4 tile; for edge tiles the caller passes a temporary C11.
// Packed A11 is stored by columns with leading dimension PACKMR = 4 and
// packed B11 by rows with leading dimension PACKNR = 4 (no broadcast,
// BBM = BBN = 1). Packing stores the inverse of each diagonal element when
// trsm preinversion is enabled (the default) and an explicit 1 for a unit
// diagonal. Only the stored triangle and the diagonal of A11 are read.
//
// Each row of the tile is held in two f64x2 vectors, so the substitution
// works on whole rows: row i is reduced by a(i,l) times each solved row l,
// summed in increasing l like the reference kernel, and then scaled by the
// diagonal. The tile and packing sizes must change together with those of
// the gemm kernel.

#define DTRSM_MR      4
#define DTRSM_NR      4
#define DTRSM_NV      ( DTRSM_NR / 2 )   // f64x2 vectors per row
#define DTRSM_PACKMR  DTRSM_MR
#define DTRSM_PACKNR  DTRSM_NR

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
#define DTRSM_DIAG( x, d ) wasm_f64x2_mul( x, d )
#else
#define DTRSM_DIAG( x, d ) wasm_f64x2_div( x, d )
#endif

static inline void dtrsm_load( const double* b, v128_t x[ DTRSM_MR ][ DTRSM_NV ] )
{
	for ( int i = 0; i < DTRSM_MR; ++i )
	for ( int v = 0; v < DTRSM_NV; ++v )
		x[ i ][ v ] = wasm_v128_load( b + i*DTRSM_PACKNR + 2*v );
}

// x[ i ] := ( x[ i ] - sum_{l0 <= l < l1} a(i,l) * x[ l ] ) / a(i,i)
static inline void dtrsm_row
     (
       const double* a,
             v128_t  x[ DTRSM_MR ][ DTRSM_NV ],
             int     i,
             int     l0,
             int     l1
     )
{
	if ( l0 < l1 )
	{
		v128_t rho[ DTRSM_NV ];
		const v128_t a0 = wasm_f64x2_splat( a[ i + l0*DTRSM_PACKMR ] );
		for ( int v = 0; v < DTRSM_NV; ++v )
			rho[ v ] = wasm_f64x2_mul( a0, x[ l0 ][ v ] );
		for ( int l = l0 + 1; l < l1; ++l )
		{
			const v128_t al = wasm_f64x2_splat( a[ i + l*DTRSM_PACKMR ] );
			for ( int v = 0; v < DTRSM_NV; ++v )
				rho[ v ] = wasm_f64x2_add( rho[ v ], wasm_f64x2_mul( al, x[ l ][ v ] ) );
		}
		for ( int v = 0; v < DTRSM_NV; ++v )
			x[ i ][ v ] = wasm_f64x2_sub( x[ i ][ v ], rho[ v ] );
	}

	const v128_t dii = wasm_f64x2_splat( a[ i + i*DTRSM_PACKMR ] );
	for ( int v = 0; v < DTRSM_NV; ++v )
		x[ i ][ v ] = DTRSM_DIAG( x[ i ][ v ], dii );
}

// Write the solved tile back to packed B11, where later gemmtrsm calls on
// the same panel read it, and to C11.
static inline void dtrsm_store
     (
       v128_t  x[ DTRSM_MR ][ DTRSM_NV ],
       double* b,
       double* c, inc_t rs_c, inc_t cs_c
     )
{
	for ( int i = 0; i < DTRSM_MR; ++i )
	for ( int v = 0; v < DTRSM_NV; ++v )
		wasm_v128_store( b + i*DTRSM_PACKNR + 2*v, x[ i ][ v ] );

	if ( cs_c == 1 )
	{
		// Row-stored C11, which includes the caller's temporary tile.
		for ( int i = 0; i < DTRSM_MR; ++i )
		for ( int v = 0; v < DTRSM_NV; ++v )
			wasm_v128_store( c + i*rs_c + 2*v, x[ i ][ v ] );
	}
	else if ( rs_c == 1 )
	{
		// Column-stored C11: transpose 2x2 blocks so that each store writes
		// two adjacent rows of one column.
		for ( int i = 0; i < DTRSM_MR; i += 2 )
		for ( int v = 0; v < DTRSM_NV; ++v )
		{
			double* c0 = c + i + ( 2*v     )*cs_c;
			double* c1 = c + i + ( 2*v + 1 )*cs_c;
			wasm_v128_store( c0, wasm_i64x2_shuffle( x[ i ][ v ], x[ i + 1 ][ v ], 0, 2 ) );
			wasm_v128_store( c1, wasm_i64x2_shuffle( x[ i ][ v ], x[ i + 1 ][ v ], 1, 3 ) );
		}
	}
	else
	{
		for ( int i = 0; i < DTRSM_MR; ++i )
		for ( int v = 0; v < DTRSM_NV; ++v )
		{
			c[ i*rs_c + ( 2*v     )*cs_c ] = wasm_f64x2_extract_lane( x[ i ][ v ], 0 );
			c[ i*rs_c + ( 2*v + 1 )*cs_c ] = wasm_f64x2_extract_lane( x[ i ][ v ], 1 );
		}
	}
}

void bli_dtrsm_l_wasm32_simd128_4x4
     (
       const void*      a0,
             void*      b0,
             void*      c0, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const double* a = a0;
	      double* b = b0;
	      double* c = c0;

	( void )data;
	( void )cntx;

	v128_t x[ DTRSM_MR ][ DTRSM_NV ];
	dtrsm_load( b, x );

	// Forward substitution: each row uses the solved rows above it.
	dtrsm_row( a, x, 0, 0, 0 );
	dtrsm_row( a, x, 1, 0, 1 );
	dtrsm_row( a, x, 2, 0, 2 );
	dtrsm_row( a, x, 3, 0, 3 );

	dtrsm_store( x, b, c, rs_c, cs_c );
}

void bli_dtrsm_u_wasm32_simd128_4x4
     (
       const void*      a0,
             void*      b0,
             void*      c0, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const double* a = a0;
	      double* b = b0;
	      double* c = c0;

	( void )data;
	( void )cntx;

	v128_t x[ DTRSM_MR ][ DTRSM_NV ];
	dtrsm_load( b, x );

	// Backward substitution: each row uses the solved rows below it.
	dtrsm_row( a, x, 3, 4, 4 );
	dtrsm_row( a, x, 2, 3, 4 );
	dtrsm_row( a, x, 1, 2, 4 );
	dtrsm_row( a, x, 0, 1, 4 );

	dtrsm_store( x, b, c, rs_c, cs_c );
}
