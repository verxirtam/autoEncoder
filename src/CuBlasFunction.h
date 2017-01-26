/*
 * =====================================================================================
 *
 *       Filename:  CuBlasFunction.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月19日 02時41分09秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include "DeviceVector.h"
#include "DeviceMatrix.h"

//y = alpha * x + y;
void Saxpy
	(
		const float* const alpha,
		const DeviceVector& x,
		DeviceVector& y
	);

//y = alpha * op(A) * x + beta * y;
void Sgemv
	(
		const float* alpha,
		cublasOperation_t op,
		const DeviceMatrix& A,
		const DeviceVector& x,
		const float* beta,
		DeviceVector& y
	);

//A = alpha * x * x^T + A; A : symmetric matrix
void Ssyr
	(
		const float* alpha,
		const DeviceVector& x,
		DeviceMatrix& A
	);

//C = alpha * op(A) * (op(A))^T + beta * C; C : symmetric matrix
void Ssyrk
	(
		const float* alpha,
		cublasOperation_t op,
		const DeviceMatrix& A,
		const float* beta,
		DeviceMatrix& C
	);

//C = alpha * op_A(A) + beta * op_B(B);
void Sgeam
	(
		const float* alpha,
		cublasOperation_t op_A,
		const DeviceMatrix& A,
		const float* beta,
		cublasOperation_t op_B,
		const DeviceMatrix& B,
		DeviceMatrix& C
	);

//x = alpha * x;
void Sscal
	(
		const float* alpha,
		DeviceVector& x
	);

//A = alpha * A;
void Sscal
	(
		const float* alpha,
		DeviceMatrix& A
	);

//C = A * diag(x)
void Sdgmm
	(
		const DeviceMatrix& A,
		const DeviceVector& x,
		DeviceMatrix& C
	);

//C = diag(x) * A
void Sdgmm
	(
		const DeviceVector& x,
		const DeviceMatrix& A,
		DeviceMatrix& C
	);
