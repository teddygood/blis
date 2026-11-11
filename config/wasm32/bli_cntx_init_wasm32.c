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

void bli_cntx_init_wasm32( cntx_t* cntx )
{
	// Begin with the reference context so that every unregistered entry
	// keeps its reference implementation.
	bli_cntx_init_wasm32_ref( cntx );

	// Register kernels independently of the kernel-set vector flags.
	bli_cntx_set_ukrs
	(
	  cntx,

	  BLIS_GEMM_UKR, BLIS_FLOAT,  bli_sgemm_wasm32_simd128_4x8,
	  BLIS_GEMM_UKR, BLIS_DOUBLE, bli_dgemm_wasm32_simd128_4x4,

	  BLIS_AXPYV_KER, BLIS_FLOAT,  bli_saxpyv_wasm32_simd128,
	  BLIS_AXPYV_KER, BLIS_DOUBLE, bli_daxpyv_wasm32_simd128,

	  BLIS_DOTV_KER, BLIS_FLOAT,  bli_sdotv_wasm32_simd128,
	  BLIS_DOTV_KER, BLIS_DOUBLE, bli_ddotv_wasm32_simd128,

	  BLIS_DOTXV_KER, BLIS_FLOAT,  bli_sdotxv_wasm32_simd128,
	  BLIS_DOTXV_KER, BLIS_DOUBLE, bli_ddotxv_wasm32_simd128,

	  BLIS_AXPYF_KER, BLIS_FLOAT,  bli_saxpyf_wasm32_simd128_8,
	  BLIS_AXPYF_KER, BLIS_DOUBLE, bli_daxpyf_wasm32_simd128_8,

	  BLIS_DOTXF_KER, BLIS_FLOAT,  bli_sdotxf_wasm32_simd128,
	  BLIS_DOTXF_KER, BLIS_DOUBLE, bli_ddotxf_wasm32_simd128,

	  // The reference gemmtrsm microkernel pairs these with the gemm
	  // microkernels above; each solves the same tile as the gemm kernel
	  // of its datatype (4x8 single, 4x4 double), so the two must change
	  // together.
	  BLIS_TRSM_L_UKR, BLIS_FLOAT,  bli_strsm_l_wasm32_simd128_4x8,
	  BLIS_TRSM_U_UKR, BLIS_FLOAT,  bli_strsm_u_wasm32_simd128_4x8,
	  BLIS_TRSM_L_UKR, BLIS_DOUBLE, bli_dtrsm_l_wasm32_simd128_4x4,
	  BLIS_TRSM_U_UKR, BLIS_DOUBLE, bli_dtrsm_u_wasm32_simd128_4x4,

	  BLIS_VA_END
	);

	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT,  TRUE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,

	  BLIS_VA_END
	);

	// Match the register and packing sizes to the 4x8 single and 4x4 double
	// kernels.
	blksz_t mr, nr;
	bli_blksz_init_easy( &mr, 4, 4, -1, -1 );
	bli_blksz_init_easy( &nr, 8, 4, -1, -1 );

	bli_cntx_set_blkszs
	(
	  cntx,

	  BLIS_MR, &mr, BLIS_MR,
	  BLIS_NR, &nr, BLIS_NR,

	  BLIS_VA_END
	);

#ifndef BLIS_DISABLE_SUP_HANDLING
	// Route small single-precision GEMM to the wasm32 sup handler. An MT
	// threshold of 129 reaches every problem bounded by 128 in all three
	// dimensions; the handler checks all three bounds and declines the rest.
	bli_cntx_set_blksz_def_dt( BLIS_FLOAT, BLIS_MT, 129, cntx );
	bli_cntx_set_blksz_max_dt( BLIS_FLOAT, BLIS_MT, 129, cntx );
	bli_cntx_set_l3_sup_handler
	(
	  BLIS_GEMM, ( void_fp )bli_sgemmsup_wasm32_simd128, cntx
	);
	// The threshold is shared with GEMMT. This wrapper keeps float GEMMT on
	// the conventional path and the other datatypes on bli_gemmtsup_ref().
	bli_cntx_set_l3_sup_handler
	(
	  BLIS_GEMMT, ( void_fp )bli_gemmtsup_wasm32, cntx
	);
#endif
}
