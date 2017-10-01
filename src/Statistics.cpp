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
	
	//TODO デバッグ用の出力
	std::cout << "N = " << N << std::endl;
	printVector(X_1N.get(), "X_1N");
	
	//分散共分散行列
	//TODO 算出方法が正しいか確認すること
	varCovMatrix = DeviceMatrix(D, D, std::vector<float>(D * D, 0.0f));
	//varCovMatrix = 1.0f * X * X^T + 0.0f * varCovMatrix;
	//             = 1.0f * X * X^T;
	alpha = 1.0f;
	beta  = 0.0f;
	Ssyrk(&alpha, CUBLAS_OP_N, X, &beta, varCovMatrix);
	//varCovMatrixの値は上半分のみ設定される
	
	//ストリームの完了待ち:varCovMatrixの算出待ち
	CUDA_CALL(cudaStreamSynchronize(stream));
	
	//TODO デバッグ用の出力
	printVector(varCovMatrix.get(), "X * X^T");
	
	//varCovMatrix = DeviceMatrix(D, D, std::vector<float>(D * D, 0.0f));//TODO This is for debug.
	//varCovMatrix = - (1 / N) * (X_1N) * (X_1N)^T + varCovMatrix;
	//             = - (1 / N) * (X_1N) * (X_1N)^T + X * X^T;
	//             = X * X^T - (1 / N) * (X_1N) * (X_1N)^T;
	alpha = - 1.0f / static_cast<float>(N);
	Ssyr(&alpha, X_1N, varCovMatrix);
	//varCovMatrixの値は上半分のみ設定される
	
	//ストリームの完了待ち:varCovMatrixの算出待ち
	CUDA_CALL(cudaStreamSynchronize(stream));
	
	//TODO デバッグ用の出力
	std::cout << "X_1N.dimension = " << X_1N.getDimension() << std::endl;
	printVector(varCovMatrix.get(), "X * X^T - (1 / N) * (X_1N) * (X_1N)^T");// <-----ここが違っているように見える
	
	//varCovMatrix = (1 / N) * varCovMatrix;
	//             = (1 / N) * (X * X^T -(1 / N) * (X_1N) * (X_1N)^T);
	alpha = 1.0f / static_cast<float>(N);
	Sscal(&alpha, varCovMatrix);
	//この時点で分散共分散行列varCovMatrixが取得できた
	//ただしvarCovMatrixの値は上半分のみ設定されている
	
	//TODO デバッグ用の出力
	printVector(varCovMatrix.get(), "varCovMatrix");
	
}






