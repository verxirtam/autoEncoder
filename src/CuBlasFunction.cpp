/*
 * =====================================================================================
 *
 *       Filename:  CUBLASFunction.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月19日 04時18分10秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "CuBlasFunction.h"

//y = alpha * x + y;
void Saxpy
	(
		const float* const alpha,
		const DeviceVector& x,
		DeviceVector& y
	)
{
	int N = x.getDimension();
	CUBLAS_CALL(cublasSaxpy(CuBlasManager::getHandle(), N, alpha, x.getAddress(), 1, y.getAddress(), 1));
}

//y = alpha * op(A) * x + beta * y;
void Sgemv
	(
		const float* alpha,
		cublasOperation_t op,
		const DeviceMatrix& A,
		const DeviceVector& x,
		const float* beta,
		DeviceVector& y
	)
{
	int M = A.getRowCount();
	int N = A.getColumnCount();
	CUBLAS_CALL
		(
			cublasSgemv
				(
					CuBlasManager::getHandle(), op, M, N,
					alpha, A.getAddress(), M,
					x.getAddress(), 1,
					beta, y.getAddress(), 1
				)
		);
}

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
	)
{
	int M = A.getRowCount();
	int N = A.getColumnCount();
	CUBLAS_CALL
		(
			cublasSgeam
				(
				 CuBlasManager::getHandle(), op_A, op_B, M, N, 
				 alpha, A.getAddress(), M,
				 beta, B.getAddress(), M,
				 C.getAddress(), M
				)
		);
}


//x = alpha * x;
void Sscal
	(
		const float* alpha,
		DeviceVector& x
	)
{
	int N = x.getDimension();
	CUBLAS_CALL
		(
			cublasSscal
				(
					CuBlasManager::getHandle(),
					N,
					alpha,
					x.getAddress(),
					1
				)
		);
}

//A = alpha * A;
void Sscal
	(
		const float* alpha,
		DeviceMatrix& A
	)
{
	int M = A.getRowCount();
	int N = A.getColumnCount();
	//行列Aを(M * N)次元ベクトルとみなしてスカラー倍する
	try
	{
		CUBLAS_CALL
			(
				cublasSscal
					(
						CuBlasManager::getHandle(),
						M * N,
						alpha,
						A.getAddress(),
						1
					)
			);
	}
	catch(CuBlasException& e)
	{
		std::stringstream msg;
		msg << e.what();
		msg << " at ";
		msg << __FILE__ << ":";
		msg << __LINE__ << " ";
		msg << "CuBlasManager::getHandle() = " << CuBlasManager::getHandle() << ", ";
		msg << "M = " << M << ", ";
		msg << "N = " << N << ", ";
		msg << "*alpha = " << *alpha << ", ";
		msg << "A.getAddress() = " << A.getAddress() << ", ";
		throw CuBlasException(msg.str());
	}
}

