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

// Fortran-77 BLAS ABI probe for wasm32.
//
// WebAssembly checks function signatures: when a caller's prototype does
// not match the definition, wasm-ld only warns and replaces the call with
// a trap. This file declares the F77 symbols the way an external caller
// does, without including blis.h, and must be linked against libblis.a
// with -Wl,--fatal-warnings so that a signature mismatch fails the link.
// A WebAssembly signature records only the argument count, the return
// type and the types of arguments passed by value. Every pointer is an
// i32 on wasm32, so a wrong pointee type (int * against int64_t *, for
// example) still links; the known-answer checks below catch that.
//
// The declarations follow SciPy's generated _blas_subroutines.h for an
// LP64 build (void subroutines, no hidden string lengths, two-argument
// lsame_) and its dummy G77 ABI wrappers for complex-valued functions.
// They cover the routines whose F77 signatures pyodide-2.1 normalised
// (banded and packed level-2, Givens rotations, lsame_, xerbla_array_)
// plus a few common entry points. Each call is checked on a small case
// with a known answer.
//
// Building with -DF77_ABI_NEGATIVE_CONTROL declares dgbmv_ with the old
// f2c int return instead. That link must fail; the CI runner checks this
// to show the probe can detect a mismatch.

#include <complex.h>
#include <math.h>
#include <stdio.h>

void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);
void zgemm_(char *transa, char *transb, int *m, int *n, int *k, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *b, int *ldb, double _Complex *beta, double _Complex *c, int *ldc);
void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
#ifdef F77_ABI_NEGATIVE_CONTROL
int dgbmv_(char *trans, int *m, int *n, int *kl, int *ku, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
#else
void dgbmv_(char *trans, int *m, int *n, int *kl, int *ku, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
#endif
void dsbmv_(char *uplo, int *n, int *k, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
void dspmv_(char *uplo, int *n, double *alpha, double *ap, double *x, int *incx, double *beta, double *y, int *incy);
void dspr_(char *uplo, int *n, double *alpha, double *x, int *incx, double *ap);
void dspr2_(char *uplo, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *ap);
void dtbmv_(char *uplo, char *trans, char *diag, int *n, int *k, double *a, int *lda, double *x, int *incx);
void dtbsv_(char *uplo, char *trans, char *diag, int *n, int *k, double *a, int *lda, double *x, int *incx);
void dtpmv_(char *uplo, char *trans, char *diag, int *n, double *ap, double *x, int *incx);
void dtpsv_(char *uplo, char *trans, char *diag, int *n, double *ap, double *x, int *incx);
void zhbmv_(char *uplo, int *n, int *k, double _Complex *alpha, double _Complex *a, int *lda, double _Complex *x, int *incx, double _Complex *beta, double _Complex *y, int *incy);
void zhpmv_(char *uplo, int *n, double _Complex *alpha, double _Complex *ap, double _Complex *x, int *incx, double _Complex *beta, double _Complex *y, int *incy);
void zhpr_(char *uplo, int *n, double *alpha, double _Complex *x, int *incx, double _Complex *ap);
void zhpr2_(char *uplo, int *n, double _Complex *alpha, double _Complex *x, int *incx, double _Complex *y, int *incy, double _Complex *ap);
void drot_(int *n, double *dx, int *incx, double *dy, int *incy, double *c, double *s);
void drotg_(double *a, double *b, double *c, double *s);
void drotm_(int *n, double *dx, int *incx, double *dy, int *incy, double *dparam);
void drotmg_(double *dd1, double *dd2, double *dx1, double *dy1, double *dparam);
void crotg_(float _Complex *a, float _Complex *b, float *c, float _Complex *s);
void zrotg_(double _Complex *a, double _Complex *b, double *c, double _Complex *s);
double ddot_(int *n, double *dx, int *incx, double *dy, int *incy);
float sdot_(int *n, float *sx, int *incx, float *sy, int *incy);
double dnrm2_(int *n, double *x, int *incx);
int idamax_(int *n, double *dx, int *incx);
double dcabs1_(double _Complex *z);
float _Complex cdotc_(int *n, float _Complex *cx, int *incx, float _Complex *cy, int *incy);
double _Complex zdotu_(int *n, double _Complex *zx, int *incx, double _Complex *zy, int *incy);
int lsame_(char *ca, char *cb);
void xerbla_array_(char *srname_array, int *srname_len, int *info);

static int n_check, n_fail;

static void check( int ok, const char* what )
{
	++n_check;
	if ( !ok ) { ++n_fail; fprintf( stderr, "FAIL: %s\n", what ); }
}

static int near( double got, double want, double tol )
{
	return isfinite( got ) && fabs( got - want ) <= tol;
}

static int znear( double _Complex got, double _Complex want, double tol )
{
	return near( creal( got ), creal( want ), tol ) &&
	       near( cimag( got ), cimag( want ), tol );
}

static int vec_eq( const double* got, const double* want, int n )
{
	for ( int i = 0; i < n; ++i ) if ( got[i] != want[i] ) return 0;
	return 1;
}

int main( void )
{
	char cn = 'N', cu = 'U';
	int  one = 1, two = 2, three = 3;

	// Level 3.
	{
		double a[4] = { 1, 3, 2, 4 }, b[4] = { 5, 7, 6, 8 }, c[4] = { -1, -1, -1, -1 };
		double alpha = 1, beta = 0, want[4] = { 19, 43, 22, 50 };
		dgemm_( &cn, &cn, &two, &two, &two, &alpha, a, &two, b, &two, &beta, c, &two );
		check( vec_eq( c, want, 4 ), "dgemm_" );
	}
	{
		float a[4] = { 1, 3, 2, 4 }, b[4] = { 5, 7, 6, 8 }, c[4] = { -1, -1, -1, -1 };
		float alpha = 1, beta = 0;
		sgemm_( &cn, &cn, &two, &two, &two, &alpha, a, &two, b, &two, &beta, c, &two );
		check( c[0] == 19 && c[1] == 43 && c[2] == 22 && c[3] == 50, "sgemm_" );
	}
	{
		double _Complex a = 1 + 2*I, b = 3 + 4*I, c = -1, alpha = 1, beta = 0;
		zgemm_( &cn, &cn, &one, &one, &one, &alpha, &a, &one, &b, &one, &beta, &c, &one );
		check( c == -5 + 10*I, "zgemm_" );
	}

	// Level 2, including the banded and packed routines.
	{
		double a[4] = { 1, 3, 2, 4 }, x[2] = { 1, 1 }, y[2] = { -1, -1 };
		double alpha = 1, beta = 0, want[2] = { 3, 7 };
		dgemv_( &cn, &two, &two, &alpha, a, &two, x, &one, &beta, y, &one );
		check( vec_eq( y, want, 2 ), "dgemv_" );
	}
	{
		// Tridiagonal 3x3 with 2 on the diagonal and -1 off it, kl = ku = 1.
		double a[9] = { 0, 2, -1,  -1, 2, -1,  -1, 2, 0 };
		double x[3] = { 1, 2, 3 }, y[3] = { -1, -1, -1 };
		double alpha = 1, beta = 0, want[3] = { 0, 0, 4 };
		dgbmv_( &cn, &three, &three, &one, &one, &alpha, a, &three,
		        x, &one, &beta, y, &one );
		check( vec_eq( y, want, 3 ), "dgbmv_" );
	}
	{
		// Symmetric [[2, 1], [1, 3]], upper band storage with k = 1.
		double a[4] = { 0, 2, 1, 3 }, x[2] = { 1, 1 }, y[2] = { -1, -1 };
		double alpha = 1, beta = 0, want[2] = { 3, 4 };
		dsbmv_( &cu, &two, &one, &alpha, a, &two, x, &one, &beta, y, &one );
		check( vec_eq( y, want, 2 ), "dsbmv_" );
	}
	{
		double ap[3] = { 1, 2, 3 }, x[2] = { 1, 1 }, y[2] = { -1, -1 };
		double alpha = 1, beta = 0, want[2] = { 3, 5 };
		dspmv_( &cu, &two, &alpha, ap, x, &one, &beta, y, &one );
		check( vec_eq( y, want, 2 ), "dspmv_" );
	}
	{
		double ap[3] = { 0, 0, 0 }, x[2] = { 1, 2 }, alpha = 1, want[3] = { 1, 2, 4 };
		dspr_( &cu, &two, &alpha, x, &one, ap );
		check( vec_eq( ap, want, 3 ), "dspr_" );
	}
	{
		double ap[3] = { 0, 0, 0 }, x[2] = { 1, 0 }, y[2] = { 0, 1 };
		double alpha = 1, want[3] = { 0, 1, 0 };
		dspr2_( &cu, &two, &alpha, x, &one, y, &one, ap );
		check( vec_eq( ap, want, 3 ), "dspr2_" );
	}
	{
		// Upper triangular [[1, 2], [0, 3]] in band (k = 1) and packed form.
		double ab[4] = { 0, 1, 2, 3 }, ap[3] = { 1, 2, 3 };
		double x[2] = { 1, 1 }, mv[2] = { 3, 3 }, sv[2] = { 1, 1 };
		dtbmv_( &cu, &cn, &cn, &two, &one, ab, &two, x, &one );
		check( vec_eq( x, mv, 2 ), "dtbmv_" );
		dtbsv_( &cu, &cn, &cn, &two, &one, ab, &two, x, &one );
		check( vec_eq( x, sv, 2 ), "dtbsv_" );
		dtpmv_( &cu, &cn, &cn, &two, ap, x, &one );
		check( vec_eq( x, mv, 2 ), "dtpmv_" );
		dtpsv_( &cu, &cn, &cn, &two, ap, x, &one );
		check( vec_eq( x, sv, 2 ), "dtpsv_" );
	}
	{
		// Hermitian [[2, 1+i], [1-i, 3]] in upper band (k = 1) and packed form.
		double _Complex ab[4] = { 0, 2, 1 + I, 3 }, ap[3] = { 2, 1 + I, 3 };
		double _Complex x0[2] = { 1, 0 }, x1[2] = { 0, 1 }, y[2] = { -1, -1 };
		double _Complex alpha = 1, beta = 0;
		zhbmv_( &cu, &two, &one, &alpha, ab, &two, x0, &one, &beta, y, &one );
		check( y[0] == 2 && y[1] == 1 - I, "zhbmv_" );
		zhpmv_( &cu, &two, &alpha, ap, x1, &one, &beta, y, &one );
		check( y[0] == 1 + I && y[1] == 3, "zhpmv_" );
	}
	{
		double _Complex ap[3] = { 0, 0, 0 }, x[2] = { 1, I };
		double alpha = 1;
		zhpr_( &cu, &two, &alpha, x, &one, ap );
		check( ap[0] == 1 && ap[1] == -I && ap[2] == 1, "zhpr_" );
	}
	{
		double _Complex ap[3] = { 0, 0, 0 }, x[2] = { 1, 0 }, y[2] = { 0, I };
		double _Complex alpha = 1;
		zhpr2_( &cu, &two, &alpha, x, &one, y, &one, ap );
		check( ap[0] == 0 && ap[1] == -I && ap[2] == 0, "zhpr2_" );
	}

	// Level 1: Givens rotations and value-returning functions.
	{
		double x[2] = { 1, 0 }, y[2] = { 0, 1 }, c = 0.6, s = 0.8;
		drot_( &two, x, &one, y, &one, &c, &s );
		check( near( x[0], 0.6, 1e-15 ) && near( x[1], 0.8, 1e-15 ) &&
		       near( y[0], -0.8, 1e-15 ) && near( y[1], 0.6, 1e-15 ), "drot_" );
	}
	{
		double a = 3, b = 4, c = 0, s = 0;
		drotg_( &a, &b, &c, &s );
		check( near( a, 5, 1e-14 ) && near( c, 0.6, 1e-15 ) &&
		       near( s, 0.8, 1e-15 ), "drotg_" );
	}
	{
		// Flag 0: H = [[1, h12], [h21, 1]] with h21 = 2, h12 = 3.
		double x = 1, y = 1, param[5] = { 0, 0, 2, 3, 0 };
		drotm_( &one, &x, &one, &y, &one, param );
		check( x == 4 && y == 3, "drotm_" );
	}
	{
		// The modified rotation from drotmg_ must zero the second component.
		double d1 = 1, d2 = 1, x1 = 1, y1 = 1, param[5] = { 0, 0, 0, 0, 0 };
		drotmg_( &d1, &d2, &x1, &y1, param );
		double x = 1, y = 1;
		drotm_( &one, &x, &one, &y, &one, param );
		check( ( param[0] == -2 || param[0] == -1 || param[0] == 0 || param[0] == 1 ) &&
		       near( y, 0, 1e-15 ), "drotmg_" );
	}
	{
		float _Complex a = 3, b = 4, s = 0;
		float c = 0;
		crotg_( &a, &b, &c, &s );
		check( znear( a, 5, 1e-5 ) && near( c, 0.6, 1e-6 ) && znear( s, 0.8, 1e-6 ),
		       "crotg_" );
	}
	{
		double _Complex a = 3, b = 4, s = 0;
		double c = 0;
		zrotg_( &a, &b, &c, &s );
		check( znear( a, 5, 1e-14 ) && near( c, 0.6, 1e-15 ) && znear( s, 0.8, 1e-15 ),
		       "zrotg_" );
	}
	{
		double x[3] = { 1, 2, 3 }, y[3] = { 4, 5, 6 };
		float sx[3] = { 1, 2, 3 }, sy[3] = { 4, 5, 6 };
		double n2[2] = { 3, 4 }, am[3] = { 1, -5, 3 };
		double _Complex z = 3 - 4*I;
		check( ddot_( &three, x, &one, y, &one ) == 32, "ddot_" );
		check( sdot_( &three, sx, &one, sy, &one ) == 32, "sdot_" );
		check( near( dnrm2_( &two, n2, &one ), 5, 1e-15 ), "dnrm2_" );
		check( idamax_( &three, am, &one ) == 2, "idamax_" );
		check( dcabs1_( &z ) == 7, "dcabs1_" );
	}
	{
		float _Complex cx = 1 + 2*I, cy = 3 + 4*I;
		double _Complex zx = 1 + 2*I, zy = 3 + 4*I;
		check( cdotc_( &one, &cx, &one, &cy, &one ) == 11 - 2*I, "cdotc_" );
		check( zdotu_( &one, &zx, &one, &zy, &one ) == -5 + 10*I, "zdotu_" );
	}

	// Auxiliary routines.
	{
		char lo = 'a', up = 'A', other = 'b';
		check( lsame_( &lo, &up ) && !lsame_( &lo, &other ), "lsame_" );

		// Prints the standard illegal-parameter message and returns.
		char name[] = "DGEMM";
		int len = 5, info = 3;
		xerbla_array_( name, &len, &info );
	}

	printf( "test-f77-abi: %d checks, %d failures\n", n_check, n_fail );
	return n_fail ? 1 : 0;
}
