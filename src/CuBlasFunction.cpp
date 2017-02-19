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

//Y = alpha * X + Y;
void Saxpy
	(
		const float* const alpha,
		const DeviceMatrix& X,
		DeviceMatrix& Y
	)
{
	int M = X.getRowCount();
	int N = X.getColumnCount();
	CUBLAS_CALL(cublasSaxpy(CuBlasManager::getHandle(), M * N, alpha, X.getAddress(), 1, Y.getAddress(), 1));
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

//A = alpha * x * y^T + A;
void Sger
	(
		const float* alpha,
		const DeviceVector& x,
		const DeviceVector& y,
		DeviceMatrix& A
	)
{
	int M = A.getRowCount();
	int N = A.getColumnCount();
	CUBLAS_CALL
		(
			cublasSger
				(
					CuBlasManager::getHandle(), M, N,
					alpha,
					x.getAddress(), 1,
					y.getAddress(), 1,
					A.getAddress(), M
				)
		);
	
}

//A = alpha * x * x^T + A; A : symmetric matrix
void Ssyr
	(
		const float* alpha,
		const DeviceVector& x,
		DeviceMatrix& A
	)
{
	int N = A.getRowCount();
	CUBLAS_CALL
		(
			cublasSsyr
				(
					CuBlasManager::getHandle(),
					CUBLAS_FILL_MODE_UPPER,
					N,
					alpha,
					x.getAddress(),
					1,
					A.getAddress(),
					N
				)
		)
}

//C = alpha * op(A) * (op(A))^T + beta * C; C : symmetric matrix
void Ssyrk
	(
		const float* alpha,
		cublasOperation_t op,
		const DeviceMatrix& A,
		const float* beta,
		DeviceMatrix& C
	)
{
	int N = A.getRowCount();
	int K = A.getColumnCount();
	
	CUBLAS_CALL
		(
			cublasSsyrk
				(
					CuBlasManager::getHandle(),
					CUBLAS_FILL_MODE_UPPER,
					op,
					N, K,
					alpha,
					A.getAddress(),
					N,
					beta,
					C.getAddress(),
					N
				)
		);
}

//C = alpha * op_A(A) * op_B(B) + beta * C;
void Sgemm
	(
		const float* alpha,
		cublasOperation_t op_A,
		const DeviceMatrix& A,
		cublasOperation_t op_B,
		const DeviceMatrix& B,
		const float* beta,
		DeviceMatrix& C
	)
{
	int M = C.getRowCount();
	int N = C.getColumnCount();
	int K = (op_A == CUBLAS_OP_N) ? A.getColumnCount() : A.getRowCount();
	CUBLAS_CALL
		(
			cublasSgemm
				(
				 CuBlasManager::getHandle(), op_A, op_B, M, N, K,
				 alpha, A.getAddress(), A.getRowCount(),
				        B.getAddress(), B.getRowCount(),
				 beta,  C.getAddress(), C.getRowCount()
				)
		);
}

//C = alpha * A * B + beta * C; A : symmetric matrix
void Ssymm
	(
		const float* alpha,
		const DeviceMatrix& A,
		const DeviceMatrix& B,
		const float* beta,
		DeviceMatrix& C
	)
{
	int M = C.getRowCount();
	int N = C.getColumnCount();
	CUBLAS_CALL
		(
			cublasSsymm
				(
					CuBlasManager::getHandle(),
					CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
					M, N,
					alpha,
					A.getAddress(), M,
					B.getAddress(), M,
					beta,
					C.getAddress(), M
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

//C = A * diag(x)
void Sdgmm
	(
		const DeviceMatrix& A,
		const DeviceVector& x,
		DeviceMatrix& C
	)
{
	int M = A.getRowCount();
	int N = A.getColumnCount();
	
	CUBLAS_CALL
		(
			cublasSdgmm
				(
					CuBlasManager::getHandle(),
					CUBLAS_SIDE_RIGHT,
					M, N,
					A.getAddress(), M,
					x.getAddress(), 1,
					C.getAddress(), M
				)
		);
}


//C = diag(x) * A
void Sdgmm
	(
		const DeviceVector& x,
		const DeviceMatrix& A,
		DeviceMatrix& C
	)
{
	int M = A.getRowCount();
	int N = A.getColumnCount();
	
	CUBLAS_CALL
		(
			cublasSdgmm
				(
					CuBlasManager::getHandle(),
					CUBLAS_SIDE_LEFT,
					M, N,
					A.getAddress(), M,
					x.getAddress(), 1,
					C.getAddress(), M
				)
		);
}


