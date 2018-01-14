/*
 * =====================================================================================
 *
 *       Filename:  Statistics.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年10月01日 05時45分56秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "Statistics.h"

//名前空間cudaを使用
using namespace cuda;

namespace nn
{

//平均と分散共分散行列を求める
void getMeanAndVarCovMatrix(const DeviceMatrix& sample, DeviceVector& mean, DeviceMatrix& varCovMatrix, cudaStream_t stream)
{
	//変数の読み替え
	const DeviceMatrix& X = sample;
	
	//データの次元
	unsigned int D = X.getRowCount();
	//データ数
	unsigned int N = X.getColumnCount();
	//全成分が1のベクトル
	auto _1N = DeviceVector::get1Vector(N);
	//X * _1N
	DeviceVector X_1N(D);
	//X_1N = 1.0f * X * _1N;
	float alpha = 1.0f;
	float beta = 0.0f;
	Sgemv(&alpha, CUBLAS_OP_N, X, _1N, &beta, X_1N);
	
	//ストリームの完了待ち:X_1Nの算出待ち
	CUDA_CALL(cudaStreamSynchronize(stream));
	
	//平均
	alpha = 1.0f / static_cast<float>(N);
	mean = X_1N;
	Sscal(&alpha, mean);
	
	//分散共分散行列
	// X_ = X - mean * _1N^T;
	// varCovMatrix = (1 / N) * X_ * X_^T
	
	// X_ = X - mean * _1N^T;
	//		X_ = X;
	//		Sger(-1.0f, mean, _1N, X_);
	//			<=>
	//			X_ = (-1.0f) * mean * _1N^T + X;
	//			   = X - mean * _1N^T;
	DeviceMatrix X_ = X;
	alpha = - 1.0f;
	Sger(&alpha, mean, _1N, X_);
	
	//ストリームの完了待ち:X_の算出待ち
	CUDA_CALL(cudaStreamSynchronize(stream));
	
	// varCovMatrix = (1 / N) * X_ * X_^T
	//              = (1 / N) * X_ * X_^T + 0.0f * varCovMatrix
	//		Ssyrk((1 / N), CUBLAS_OP_N, X_, 0.0f, varCovMatrix);
	varCovMatrix = DeviceMatrix(D, D);
	alpha = 1.0f / static_cast<float>(N);
	beta  = 0.0f;
	Ssyrk(&alpha, CUBLAS_OP_N, X_, &beta, varCovMatrix);
	
}


}



