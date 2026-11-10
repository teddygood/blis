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

// Link-only check of every Fortran BLAS prototype that SciPy declares.
//
// Generated from scipy/linalg/cython_blas_signatures.txt and the
// xerbla_array entry of scipy/linalg/cython_lapack_signatures.txt at
// SciPy commit 2bc4099d1e83526edfab71b8af1c88b72af623d1, using SciPy's C
// types for an LP64 build: int and bint become int, and s, d, c and z
// become float, double, float _Complex and double _Complex. SciPy calls
// cdotc, cdotu, zdotc and zdotu through the wrappers in
// scipy/_build_utils/src/wrap_dummy_g77_abi.c, which declare them as
// returning the complex value; this file does the same.
//
// wasm-ld checks a caller's prototype only at a direct call: a file that
// just takes the functions' addresses links even with a wrong prototype.
// So link_all() calls every function directly, and main() never runs it.
// Linked with -Wl,--fatal-warnings, any mismatch in the WebAssembly
// signature (argument count, return type, types of arguments passed by
// value) fails the link. Every pointer is an i32 on wasm32, so a wrong
// pointee type is not seen here; test-f77-abi.c runs a subset of these
// routines on known answers.
//
// Building with -DF77_ABI_NEGATIVE_CONTROL declares zdrot_ with the old
// f2c int return instead. That link must fail; the CI runner checks this.

#include <stdio.h>

void caxpy_(int *n, float _Complex *ca, float _Complex *cx, int *incx, float _Complex *cy, int *incy);
void ccopy_(int *n, float _Complex *cx, int *incx, float _Complex *cy, int *incy);
float _Complex cdotc_(int *n, float _Complex *cx, int *incx, float _Complex *cy, int *incy);
float _Complex cdotu_(int *n, float _Complex *cx, int *incx, float _Complex *cy, int *incy);
void cgbmv_(char *trans, int *m, int *n, int *kl, int *ku, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *x, int *incx, float _Complex *beta, float _Complex *y, int *incy);
void cgemm_(char *transa, char *transb, int *m, int *n, int *k, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *b, int *ldb, float _Complex *beta, float _Complex *c, int *ldc);
void cgemv_(char *trans, int *m, int *n, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *x, int *incx, float _Complex *beta, float _Complex *y, int *incy);
void cgerc_(int *m, int *n, float _Complex *alpha, float _Complex *x, int *incx, float _Complex *y, int *incy, float _Complex *a, int *lda);
void cgeru_(int *m, int *n, float _Complex *alpha, float _Complex *x, int *incx, float _Complex *y, int *incy, float _Complex *a, int *lda);
void chbmv_(char *uplo, int *n, int *k, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *x, int *incx, float _Complex *beta, float _Complex *y, int *incy);
void chemm_(char *side, char *uplo, int *m, int *n, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *b, int *ldb, float _Complex *beta, float _Complex *c, int *ldc);
void chemv_(char *uplo, int *n, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *x, int *incx, float _Complex *beta, float _Complex *y, int *incy);
void cher_(char *uplo, int *n, float *alpha, float _Complex *x, int *incx, float _Complex *a, int *lda);
void cher2_(char *uplo, int *n, float _Complex *alpha, float _Complex *x, int *incx, float _Complex *y, int *incy, float _Complex *a, int *lda);
void cher2k_(char *uplo, char *trans, int *n, int *k, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *b, int *ldb, float *beta, float _Complex *c, int *ldc);
void cherk_(char *uplo, char *trans, int *n, int *k, float *alpha, float _Complex *a, int *lda, float *beta, float _Complex *c, int *ldc);
void chpmv_(char *uplo, int *n, float _Complex *alpha, float _Complex *ap, float _Complex *x, int *incx, float _Complex *beta, float _Complex *y, int *incy);
void chpr_(char *uplo, int *n, float *alpha, float _Complex *x, int *incx, float _Complex *ap);
void chpr2_(char *uplo, int *n, float _Complex *alpha, float _Complex *x, int *incx, float _Complex *y, int *incy, float _Complex *ap);
void crotg_(float _Complex *a, float _Complex *b, float *c, float _Complex *s);
void cscal_(int *n, float _Complex *ca, float _Complex *cx, int *incx);
void csrot_(int *n, float _Complex *cx, int *incx, float _Complex *cy, int *incy, float *c, float *s);
void csscal_(int *n, float *sa, float _Complex *cx, int *incx);
void cswap_(int *n, float _Complex *cx, int *incx, float _Complex *cy, int *incy);
void csymm_(char *side, char *uplo, int *m, int *n, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *b, int *ldb, float _Complex *beta, float _Complex *c, int *ldc);
void csyr2k_(char *uplo, char *trans, int *n, int *k, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *b, int *ldb, float _Complex *beta, float _Complex *c, int *ldc);
void csyrk_(char *uplo, char *trans, int *n, int *k, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *beta, float _Complex *c, int *ldc);
void ctbmv_(char *uplo, char *trans, char *diag, int *n, int *k, float _Complex *a, int *lda, float _Complex *x, int *incx);
void ctbsv_(char *uplo, char *trans, char *diag, int *n, int *k, float _Complex *a, int *lda, float _Complex *x, int *incx);
void ctpmv_(char *uplo, char *trans, char *diag, int *n, float _Complex *ap, float _Complex *x, int *incx);
void ctpsv_(char *uplo, char *trans, char *diag, int *n, float _Complex *ap, float _Complex *x, int *incx);
void ctrmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *b, int *ldb);
void ctrmv_(char *uplo, char *trans, char *diag, int *n, float _Complex *a, int *lda, float _Complex *x, int *incx);
void ctrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, float _Complex *alpha, float _Complex *a, int *lda, float _Complex *b, int *ldb);
void ctrsv_(char *uplo, char *trans, char *diag, int *n, float _Complex *a, int *lda, float _Complex *x, int *incx);
double dasum_(int *n, double *dx, int *incx);
void daxpy_(int *n, double *da, double *dx, int *incx, double *dy, int *incy);
double dcabs1_(double _Complex *z);
void dcopy_(int *n, double *dx, int *incx, double *dy, int *incy);
double ddot_(int *n, double *dx, int *incx, double *dy, int *incy);
void dgbmv_(char *trans, int *m, int *n, int *kl, int *ku, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *a, int *lda);
double dnrm2_(int *n, double *x, int *incx);
void drot_(int *n, double *dx, int *incx, double *dy, int *incy, double *c, double *s);
void drotg_(double *a, double *b, double *c, double *s);
void drotm_(int *n, double *dx, int *incx, double *dy, int *incy, double *dparam);
void drotmg_(double *dd1, double *dd2, double *dx1, double *dy1, double *dparam);
void dsbmv_(char *uplo, int *n, int *k, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
void dscal_(int *n, double *da, double *dx, int *incx);
double dsdot_(int *n, float *sx, int *incx, float *sy, int *incy);
void dspmv_(char *uplo, int *n, double *alpha, double *ap, double *x, int *incx, double *beta, double *y, int *incy);
void dspr_(char *uplo, int *n, double *alpha, double *x, int *incx, double *ap);
void dspr2_(char *uplo, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *ap);
void dswap_(int *n, double *dx, int *incx, double *dy, int *incy);
void dsymm_(char *side, char *uplo, int *m, int *n, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
void dsymv_(char *uplo, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
void dsyr_(char *uplo, int *n, double *alpha, double *x, int *incx, double *a, int *lda);
void dsyr2_(char *uplo, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *a, int *lda);
void dsyr2k_(char *uplo, char *trans, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
void dsyrk_(char *uplo, char *trans, int *n, int *k, double *alpha, double *a, int *lda, double *beta, double *c, int *ldc);
void dtbmv_(char *uplo, char *trans, char *diag, int *n, int *k, double *a, int *lda, double *x, int *incx);
void dtbsv_(char *uplo, char *trans, char *diag, int *n, int *k, double *a, int *lda, double *x, int *incx);
void dtpmv_(char *uplo, char *trans, char *diag, int *n, double *ap, double *x, int *incx);
void dtpsv_(char *uplo, char *trans, char *diag, int *n, double *ap, double *x, int *incx);
void dtrmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *a, int *lda, double *b, int *ldb);
void dtrmv_(char *uplo, char *trans, char *diag, int *n, double *a, int *lda, double *x, int *incx);
void dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *a, int *lda, double *b, int *ldb);
void dtrsv_(char *uplo, char *trans, char *diag, int *n, double *a, int *lda, double *x, int *incx);
double dzasum_(int *n, double _Complex *zx, int *incx);
double dznrm2_(int *n, double _Complex *x, int *incx);
int icamax_(int *n, float _Complex *cx, int *incx);
int idamax_(int *n, double *dx, int *incx);
int isamax_(int *n, float *sx, int *incx);
int izamax_(int *n, double _Complex *zx, int *incx);
int lsame_(char *ca, char *cb);
float sasum_(int *n, float *sx, int *incx);
void saxpy_(int *n, float *sa, float *sx, int *incx, float *sy, int *incy);
float scasum_(int *n, float _Complex *cx, int *incx);
float scnrm2_(int *n, float _Complex *x, int *incx);
void scopy_(int *n, float *sx, int *incx, float *sy, int *incy);
float sdot_(int *n, float *sx, int *incx, float *sy, int *incy);
float sdsdot_(int *n, float *sb, float *sx, int *incx, float *sy, int *incy);
void sgbmv_(char *trans, int *m, int *n, int *kl, int *ku, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);
void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
void sger_(int *m, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *a, int *lda);
float snrm2_(int *n, float *x, int *incx);
void srot_(int *n, float *sx, int *incx, float *sy, int *incy, float *c, float *s);
void srotg_(float *a, float *b, float *c, float *s);
void srotm_(int *n, float *sx, int *incx, float *sy, int *incy, float *sparam);
void srotmg_(float *sd1, float *sd2, float *sx1, float *sy1, float *sparam);
void ssbmv_(char *uplo, int *n, int *k, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
void sscal_(int *n, float *sa, float *sx, int *incx);
void sspmv_(char *uplo, int *n, float *alpha, float *ap, float *x, int *incx, float *beta, float *y, int *incy);
void sspr_(char *uplo, int *n, float *alpha, float *x, int *incx, float *ap);
void sspr2_(char *uplo, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *ap);
void sswap_(int *n, float *sx, int *incx, float *sy, int *incy);
void ssymm_(char *side, char *uplo, int *m, int *n, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);
void ssymv_(char *uplo, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
void ssyr_(char *uplo, int *n, float *alpha, float *x, int *incx, float *a, int *lda);
void ssyr2_(char *uplo, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *a, int *lda);
void ssyr2k_(char *uplo, char *trans, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);
void ssyrk_(char *uplo, char *trans, int *n, int *k, float *alpha, float *a, int *lda, float *beta, float *c, int *ldc);
void stbmv_(char *uplo, char *trans, char *diag, int *n, int *k, float *a, int *lda, float *x, int *incx);
void stbsv_(char *uplo, char *trans, char *diag, int *n, int *k, float *a, int *lda, float *x, int *incx);
void stpmv_(char *uplo, char *trans, char *diag, int *n, float *ap, float *x, int *incx);
void stpsv_(char *uplo, char *trans, char *diag, int *n, float *ap, float *x, int *incx);
void strmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, float *alpha, float *a, int *lda, float *b, int *ldb);
void strmv_(char *uplo, char *trans, char *diag, int *n, float *a, int *lda, float *x, int *incx);
void strsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, float *alpha, float *a, int *lda, float *b, int *ldb);
void strsv_(char *uplo, char *trans, char *diag, int *n, float *a, int *lda, float *x, int *incx);
void zaxpy_(int *n, double _Complex *za, double _Complex *zx, int *incx, double _Complex *zy, int *incy);
void zcopy_(int *n, double _Complex *zx, int *incx, double _Complex *zy, int *incy);
double _Complex zdotc_(int *n, double _Complex *zx, int *incx, double _Complex *zy, int *incy);
double _Complex zdotu_(int *n, double _Complex *zx, int *incx, double _Complex *zy, int *incy);
#ifdef F77_ABI_NEGATIVE_CONTROL
int zdrot_(int *n, double _Complex *zx, int *incx, double _Complex *zy, int *incy, double *c, double *s);
#else
void zdrot_(int *n, double _Complex *zx, int *incx, double _Complex *zy, int *incy, double *c, double *s);
#endif
void zdscal_(int *n, double *da, double _Complex *zx, int *incx);
void zgbmv_(char *trans, int *m, int *n, int *kl, int *ku, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *x, int *incx, double _Complex *beta, double _Complex *y, int *incy);
void zgemm_(char *transa, char *transb, int *m, int *n, int *k, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *b, int *ldb, double _Complex *beta, double _Complex *c, int *ldc);
void zgemv_(char *trans, int *m, int *n, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *x, int *incx, double _Complex *beta, double _Complex *y, int *incy);
void zgerc_(int *m, int *n, double _Complex *alpha, double _Complex *x, int *incx, double _Complex *y, int *incy, double _Complex *a, int *lda);
void zgeru_(int *m, int *n, double _Complex *alpha, double _Complex *x, int *incx, double _Complex *y, int *incy, double _Complex *a, int *lda);
void zhbmv_(char *uplo, int *n, int *k, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *x, int *incx, double _Complex *beta, double _Complex *y, int *incy);
void zhemm_(char *side, char *uplo, int *m, int *n, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *b, int *ldb, double _Complex *beta, double _Complex *c, int *ldc);
void zhemv_(char *uplo, int *n, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *x, int *incx, double _Complex *beta, double _Complex *y, int *incy);
void zher_(char *uplo, int *n, double *alpha, double _Complex *x, int *incx, double _Complex *a, int *lda);
void zher2_(char *uplo, int *n, double _Complex *alpha, double _Complex *x, int *incx, double _Complex *y, int *incy, double _Complex *a, int *lda);
void zher2k_(char *uplo, char *trans, int *n, int *k, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *b, int *ldb, double *beta, double _Complex *c, int *ldc);
void zherk_(char *uplo, char *trans, int *n, int *k, double *alpha, double _Complex *a, int *lda, double *beta, double _Complex *c, int *ldc);
void zhpmv_(char *uplo, int *n, double _Complex *alpha, double _Complex *ap, double _Complex *x, int *incx, double _Complex *beta, double _Complex *y, int *incy);
void zhpr_(char *uplo, int *n, double *alpha, double _Complex *x, int *incx, double _Complex *ap);
void zhpr2_(char *uplo, int *n, double _Complex *alpha, double _Complex *x, int *incx, double _Complex *y, int *incy, double _Complex *ap);
void zrotg_(double _Complex *a, double _Complex *b, double *c, double _Complex *s);
void zscal_(int *n, double _Complex *za, double _Complex *zx, int *incx);
void zswap_(int *n, double _Complex *zx, int *incx, double _Complex *zy, int *incy);
void zsymm_(char *side, char *uplo, int *m, int *n, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *b, int *ldb, double _Complex *beta, double _Complex *c, int *ldc);
void zsyr2k_(char *uplo, char *trans, int *n, int *k, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *b, int *ldb, double _Complex *beta, double _Complex *c, int *ldc);
void zsyrk_(char *uplo, char *trans, int *n, int *k, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *beta, double _Complex *c, int *ldc);
void ztbmv_(char *uplo, char *trans, char *diag, int *n, int *k, double _Complex *a, int *lda, double _Complex *x, int *incx);
void ztbsv_(char *uplo, char *trans, char *diag, int *n, int *k, double _Complex *a, int *lda, double _Complex *x, int *incx);
void ztpmv_(char *uplo, char *trans, char *diag, int *n, double _Complex *ap, double _Complex *x, int *incx);
void ztpsv_(char *uplo, char *trans, char *diag, int *n, double _Complex *ap, double _Complex *x, int *incx);
void ztrmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *b, int *ldb);
void ztrmv_(char *uplo, char *trans, char *diag, int *n, double _Complex *a, int *lda, double _Complex *x, int *incx);
void ztrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *b, int *ldb);
void ztrsv_(char *uplo, char *trans, char *diag, int *n, double _Complex *a, int *lda, double _Complex *x, int *incx);
void xerbla_array_(char *srname_array, int *srname_len, int *info);

static void link_all( void )
{
	caxpy_( 0, 0, 0, 0, 0, 0 );
	ccopy_( 0, 0, 0, 0, 0 );
	cdotc_( 0, 0, 0, 0, 0 );
	cdotu_( 0, 0, 0, 0, 0 );
	cgbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	cgemm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	cgemv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	cgerc_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	cgeru_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	chbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	chemm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	chemv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	cher_( 0, 0, 0, 0, 0, 0, 0 );
	cher2_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	cher2k_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	cherk_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	chpmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	chpr_( 0, 0, 0, 0, 0, 0 );
	chpr2_( 0, 0, 0, 0, 0, 0, 0, 0 );
	crotg_( 0, 0, 0, 0 );
	cscal_( 0, 0, 0, 0 );
	csrot_( 0, 0, 0, 0, 0, 0, 0 );
	csscal_( 0, 0, 0, 0 );
	cswap_( 0, 0, 0, 0, 0 );
	csymm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	csyr2k_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	csyrk_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ctbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ctbsv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ctpmv_( 0, 0, 0, 0, 0, 0, 0 );
	ctpsv_( 0, 0, 0, 0, 0, 0, 0 );
	ctrmm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ctrmv_( 0, 0, 0, 0, 0, 0, 0, 0 );
	ctrsm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ctrsv_( 0, 0, 0, 0, 0, 0, 0, 0 );
	dasum_( 0, 0, 0 );
	daxpy_( 0, 0, 0, 0, 0, 0 );
	dcabs1_( 0 );
	dcopy_( 0, 0, 0, 0, 0 );
	ddot_( 0, 0, 0, 0, 0 );
	dgbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dgemm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dgemv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dger_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dnrm2_( 0, 0, 0 );
	drot_( 0, 0, 0, 0, 0, 0, 0 );
	drotg_( 0, 0, 0, 0 );
	drotm_( 0, 0, 0, 0, 0, 0 );
	drotmg_( 0, 0, 0, 0, 0 );
	dsbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dscal_( 0, 0, 0, 0 );
	dsdot_( 0, 0, 0, 0, 0 );
	dspmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dspr_( 0, 0, 0, 0, 0, 0 );
	dspr2_( 0, 0, 0, 0, 0, 0, 0, 0 );
	dswap_( 0, 0, 0, 0, 0 );
	dsymm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dsymv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dsyr_( 0, 0, 0, 0, 0, 0, 0 );
	dsyr2_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dsyr2k_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dsyrk_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dtbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dtbsv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dtpmv_( 0, 0, 0, 0, 0, 0, 0 );
	dtpsv_( 0, 0, 0, 0, 0, 0, 0 );
	dtrmm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dtrmv_( 0, 0, 0, 0, 0, 0, 0, 0 );
	dtrsm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	dtrsv_( 0, 0, 0, 0, 0, 0, 0, 0 );
	dzasum_( 0, 0, 0 );
	dznrm2_( 0, 0, 0 );
	icamax_( 0, 0, 0 );
	idamax_( 0, 0, 0 );
	isamax_( 0, 0, 0 );
	izamax_( 0, 0, 0 );
	lsame_( 0, 0 );
	sasum_( 0, 0, 0 );
	saxpy_( 0, 0, 0, 0, 0, 0 );
	scasum_( 0, 0, 0 );
	scnrm2_( 0, 0, 0 );
	scopy_( 0, 0, 0, 0, 0 );
	sdot_( 0, 0, 0, 0, 0 );
	sdsdot_( 0, 0, 0, 0, 0, 0 );
	sgbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	sgemm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	sgemv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	sger_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	snrm2_( 0, 0, 0 );
	srot_( 0, 0, 0, 0, 0, 0, 0 );
	srotg_( 0, 0, 0, 0 );
	srotm_( 0, 0, 0, 0, 0, 0 );
	srotmg_( 0, 0, 0, 0, 0 );
	ssbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	sscal_( 0, 0, 0, 0 );
	sspmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	sspr_( 0, 0, 0, 0, 0, 0 );
	sspr2_( 0, 0, 0, 0, 0, 0, 0, 0 );
	sswap_( 0, 0, 0, 0, 0 );
	ssymm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ssymv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ssyr_( 0, 0, 0, 0, 0, 0, 0 );
	ssyr2_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ssyr2k_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ssyrk_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	stbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	stbsv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	stpmv_( 0, 0, 0, 0, 0, 0, 0 );
	stpsv_( 0, 0, 0, 0, 0, 0, 0 );
	strmm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	strmv_( 0, 0, 0, 0, 0, 0, 0, 0 );
	strsm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	strsv_( 0, 0, 0, 0, 0, 0, 0, 0 );
	zaxpy_( 0, 0, 0, 0, 0, 0 );
	zcopy_( 0, 0, 0, 0, 0 );
	zdotc_( 0, 0, 0, 0, 0 );
	zdotu_( 0, 0, 0, 0, 0 );
	zdrot_( 0, 0, 0, 0, 0, 0, 0 );
	zdscal_( 0, 0, 0, 0 );
	zgbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zgemm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zgemv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zgerc_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zgeru_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zhbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zhemm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zhemv_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zher_( 0, 0, 0, 0, 0, 0, 0 );
	zher2_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zher2k_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zherk_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zhpmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zhpr_( 0, 0, 0, 0, 0, 0 );
	zhpr2_( 0, 0, 0, 0, 0, 0, 0, 0 );
	zrotg_( 0, 0, 0, 0 );
	zscal_( 0, 0, 0, 0 );
	zswap_( 0, 0, 0, 0, 0 );
	zsymm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zsyr2k_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	zsyrk_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ztbmv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ztbsv_( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ztpmv_( 0, 0, 0, 0, 0, 0, 0 );
	ztpsv_( 0, 0, 0, 0, 0, 0, 0 );
	ztrmm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ztrmv_( 0, 0, 0, 0, 0, 0, 0, 0 );
	ztrsm_( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	ztrsv_( 0, 0, 0, 0, 0, 0, 0, 0 );
	xerbla_array_( 0, 0, 0 );
}

// Never set: the calls above are only linked, not run.
static volatile int run_calls = 0;

int main( void )
{
	if ( run_calls ) link_all();
	printf( "test-f77-abi-scipy: %d SciPy BLAS prototypes linked\n", 149 );
	return 0;
}
