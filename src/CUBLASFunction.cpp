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

#include "CUBLASFunction.h"


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
	cublasStatus_t stat;
	int M = A.getRowCount();
	int N = A.getColumnCount();
	stat = cublasSgemv
		(
			CUBLASManager::getHandle(), op, M, N,
			alpha, A.getAddress(), M,
			x.getAddress(), 1,
			beta, y.getAddress(), 1
		);
	if(stat != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "error at cublasSgemv()" << std::endl;
	}
}

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
	cublasStatus_t stat;
	int M = A.getRowCount();
	int N = A.getColumnCount();
	stat = cublasSgeam
		(
			CUBLASManager::getHandle(), op_A, op_B, M, N, 
			alpha, A.getAddress(), M,
			beta, B.getAddress(), M,
			C.getAddress(), M
		);
	if(stat != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "error at cublasSgemv()" << std::endl;
	}
}
