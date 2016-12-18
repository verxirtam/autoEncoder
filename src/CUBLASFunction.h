/*
 * =====================================================================================
 *
 *       Filename:  CUBLASFunction.h
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

void Sgemv
	(
		const float* alpha,
		cublasOperation_t op,
		const DeviceMatrix& A,
		const DeviceVector& x,
		const float* beta,
		DeviceVector& y
	);

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





