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

	// Replace only the double-precision real gemm microkernel. This file
	// is compiled without the kernel-set vector flags, so registration
	// must not depend on __wasm_simd128__.
	bli_cntx_set_ukrs
	(
	  cntx,

	  BLIS_GEMM_UKR, BLIS_DOUBLE, bli_dgemm_wasm32_simd128_4x4,

	  BLIS_VA_END
	);

	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,

	  BLIS_VA_END
	);

	// The kernel is 4x4: update the double register blocksizes (default
	// and maximum/packing sizes) to match. Other datatypes keep their
	// reference blocksizes.
	blksz_t mr, nr;
	bli_blksz_init_easy( &mr, -1, 4, -1, -1 );
	bli_blksz_init_easy( &nr, -1, 4, -1, -1 );

	bli_cntx_set_blkszs
	(
	  cntx,

	  BLIS_MR, &mr, BLIS_MR,
	  BLIS_NR, &nr, BLIS_NR,

	  BLIS_VA_END
	);
}
