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

// trsm microkernels for the 4x8 float tile of bli_sgemm_wasm32_simd128_4x8.
// The reference gemmtrsm microkernel updates B11 with that gemm kernel and
// then calls one of these to compute
//
//   B11 := inv( A11 ) * B11,  C11 := B11
//
// on a full MR x NR tile; for edge tiles the caller passes a temporary
// C11. Packed A11 is stored by columns with leading dimension
// PACKMR = MR and packed B11 by rows with leading dimension PACKNR = NR
// (no broadcast, BBM = BBN = 1). Packing stores the inverse of each
// diagonal element when trsm preinversion is enabled (the default) and an
// explicit 1 for a unit diagonal. Only the stored triangle and the
// diagonal of A11 are read.
//
// Each row of the tile is held in NR / 4 f32x4 vectors, so the
// substitution works on whole rows: row i is reduced by a(i,l) times each
// solved row l, summed in increasing l like the reference kernel, and then
// scaled by the diagonal. The code below depends on NR only through
// STRSM_NR, which follows the 4x8 gemm tile.

#define STRSM_MR      4
#define STRSM_NR      8
#define STRSM_NV      ( STRSM_NR / 4 )   // f32x4 vectors per row
#define STRSM_PACKMR  STRSM_MR
#define STRSM_PACKNR  STRSM_NR

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
#define STRSM_DIAG( x, d ) wasm_f32x4_mul( x, d )
#else
#define STRSM_DIAG( x, d ) wasm_f32x4_div( x, d )
#endif

static inline void strsm_load( const float* b, v128_t x[ STRSM_MR ][ STRSM_NV ] )
{
	for ( int i = 0; i < STRSM_MR; ++i )
	for ( int v = 0; v < STRSM_NV; ++v )
		x[ i ][ v ] = wasm_v128_load( b + i*STRSM_PACKNR + 4*v );
}

// x[ i ] := ( x[ i ] - sum_{l0 <= l < l1} a(i,l) * x[ l ] ) / a(i,i)
static inline void strsm_row
     (
       const float*  a,
             v128_t  x[ STRSM_MR ][ STRSM_NV ],
             int     i,
             int     l0,
             int     l1
     )
{
	if ( l0 < l1 )
	{
		v128_t rho[ STRSM_NV ];
		const v128_t a0 = wasm_f32x4_splat( a[ i + l0*STRSM_PACKMR ] );
		for ( int v = 0; v < STRSM_NV; ++v )
			rho[ v ] = wasm_f32x4_mul( a0, x[ l0 ][ v ] );
		for ( int l = l0 + 1; l < l1; ++l )
		{
			const v128_t al = wasm_f32x4_splat( a[ i + l*STRSM_PACKMR ] );
			for ( int v = 0; v < STRSM_NV; ++v )
				rho[ v ] = wasm_f32x4_add( rho[ v ], wasm_f32x4_mul( al, x[ l ][ v ] ) );
		}
		for ( int v = 0; v < STRSM_NV; ++v )
			x[ i ][ v ] = wasm_f32x4_sub( x[ i ][ v ], rho[ v ] );
	}

	const v128_t dii = wasm_f32x4_splat( a[ i + i*STRSM_PACKMR ] );
	for ( int v = 0; v < STRSM_NV; ++v )
		x[ i ][ v ] = STRSM_DIAG( x[ i ][ v ], dii );
}

// Write the solved tile back to packed B11, where later gemmtrsm calls on
// the same panel read it, and to C11.
static inline void strsm_store
     (
       v128_t x[ STRSM_MR ][ STRSM_NV ],
       float* b,
       float* c, inc_t rs_c, inc_t cs_c
     )
{
	for ( int i = 0; i < STRSM_MR; ++i )
	for ( int v = 0; v < STRSM_NV; ++v )
		wasm_v128_store( b + i*STRSM_PACKNR + 4*v, x[ i ][ v ] );

	if ( cs_c == 1 )
	{
		// Row-stored C11, which includes the caller's temporary tile.
		for ( int i = 0; i < STRSM_MR; ++i )
		for ( int v = 0; v < STRSM_NV; ++v )
			wasm_v128_store( c + i*rs_c + 4*v, x[ i ][ v ] );
	}
	else if ( rs_c == 1 )
	{
		// Column-stored C11: transpose each 4x4 block of the tile so that
		// each store writes the four rows of one column.
		for ( int v = 0; v < STRSM_NV; ++v )
		{
			const v128_t t0 = wasm_i32x4_shuffle( x[ 0 ][ v ], x[ 1 ][ v ], 0, 4, 1, 5 );
			const v128_t t1 = wasm_i32x4_shuffle( x[ 0 ][ v ], x[ 1 ][ v ], 2, 6, 3, 7 );
			const v128_t t2 = wasm_i32x4_shuffle( x[ 2 ][ v ], x[ 3 ][ v ], 0, 4, 1, 5 );
			const v128_t t3 = wasm_i32x4_shuffle( x[ 2 ][ v ], x[ 3 ][ v ], 2, 6, 3, 7 );
			float* cv = c + ( 4*v )*cs_c;
			wasm_v128_store( cv,          wasm_i32x4_shuffle( t0, t2, 0, 1, 4, 5 ) );
			wasm_v128_store( cv +   cs_c, wasm_i32x4_shuffle( t0, t2, 2, 3, 6, 7 ) );
			wasm_v128_store( cv + 2*cs_c, wasm_i32x4_shuffle( t1, t3, 0, 1, 4, 5 ) );
			wasm_v128_store( cv + 3*cs_c, wasm_i32x4_shuffle( t1, t3, 2, 3, 6, 7 ) );
		}
	}
	else
	{
		for ( int i = 0; i < STRSM_MR; ++i )
		for ( int v = 0; v < STRSM_NV; ++v )
		{
			float* ci = c + i*rs_c + ( 4*v )*cs_c;
			ci[ 0      ] = wasm_f32x4_extract_lane( x[ i ][ v ], 0 );
			ci[   cs_c ] = wasm_f32x4_extract_lane( x[ i ][ v ], 1 );
			ci[ 2*cs_c ] = wasm_f32x4_extract_lane( x[ i ][ v ], 2 );
			ci[ 3*cs_c ] = wasm_f32x4_extract_lane( x[ i ][ v ], 3 );
		}
	}
}

void bli_strsm_l_wasm32_simd128_4x8
     (
       const void*      a0,
             void*      b0,
             void*      c0, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const float* a = a0;
	      float* b = b0;
	      float* c = c0;

	( void )data;
	( void )cntx;

	v128_t x[ STRSM_MR ][ STRSM_NV ];
	strsm_load( b, x );

	// Forward substitution: each row uses the solved rows above it.
	strsm_row( a, x, 0, 0, 0 );
	strsm_row( a, x, 1, 0, 1 );
	strsm_row( a, x, 2, 0, 2 );
	strsm_row( a, x, 3, 0, 3 );

	strsm_store( x, b, c, rs_c, cs_c );
}

void bli_strsm_u_wasm32_simd128_4x8
     (
       const void*      a0,
             void*      b0,
             void*      c0, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const float* a = a0;
	      float* b = b0;
	      float* c = c0;

	( void )data;
	( void )cntx;

	v128_t x[ STRSM_MR ][ STRSM_NV ];
	strsm_load( b, x );

	// Backward substitution: each row uses the solved rows below it.
	strsm_row( a, x, 3, 4, 4 );
	strsm_row( a, x, 2, 3, 4 );
	strsm_row( a, x, 1, 2, 4 );
	strsm_row( a, x, 0, 1, 4 );

	strsm_store( x, b, c, rs_c, cs_c );
}
