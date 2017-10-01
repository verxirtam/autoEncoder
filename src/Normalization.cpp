/*
 * =====================================================================================
 *
 *       Filename:  Normalization.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年01月29日 19時58分28秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "Normalization.h"


//白色化を行う
DeviceMatrix Normalization::getWhitening(const DeviceMatrix& whiteningMatrix, const DeviceMatrix& X) const
{
	//TODO 使用するストリームを明示する&ストリームの完了待ちを行う
	CUBLAS_CALL(cublasSetStream(CuBlasManager::getHandle(), 0));
	
	unsigned int D = X.getRowCount();
	unsigned int N = X.getColumnCount();
	auto _1N = DeviceVector::get1Vector(N);//TODO いちいち1Vector作るのは無駄
	//whiteningMatrix * (X - mean * _1N^T)
	
	DeviceMatrix Y = X;
	//Y = X - mean * _1N^T;
	//  = (-1.0f) * mean * _1N^T + Y;
	float alpha = - 1.0f;
	Sger(&alpha, mean, _1N, Y);
	
	//Y = whiteningMatrix * Y;
	//  = whiteningMatrix * (X - mean * _1N^T);
	DeviceMatrix Z(D, N);
	alpha = 1.0f;
	float beta = 0.0f;
	Sgemm(&alpha, CUBLAS_OP_N, whiteningMatrix, CUBLAS_OP_N, Y, &beta, Z);
	
	//NULLストリームの完了待ち
	CUDA_CALL(cudaStreamSynchronize(0));
	
	return Z;
}


//正規化用のデータを元に平均・白色化の変換行列を作成する
void Normalization::init(const DeviceMatrix& X)
{
	//TODO 使用するストリームを明示する&ストリームの完了待ちを行う
	CUBLAS_CALL(cublasSetStream(CuBlasManager::getHandle(), 0));
	
	//データの次元
	unsigned int D = X.getRowCount();
	
	mean = DeviceVector(D);
	varCovMatrix = DeviceMatrix(D, D);
	getMeanAndVarCovMatrix(X, mean, varCovMatrix, 0);
	/* ------------------------------------------------------------------------
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
	
	//平均
	alpha = 1.0f / static_cast<float>(N);
	mean = X_1N;
	Sscal(&alpha, mean);
	
	
	//分散共分散行列
	//TODO 算出方法が正しいか確認すること
	//TODO 分散共分散行列の算出を外出しする
	varCovMatrix = DeviceMatrix(D,D);
	//varCovMatrix = 1.0f * X * X^T;
	alpha = 1.0f;
	beta = 0.0f;
	Ssyrk(&alpha, CUBLAS_OP_N, X, &beta, varCovMatrix);
	//varCovMatrixの値は上半分のみ設定される
	
	//varCovMatrix = -(1 / N) * (X_1N) * (X_1N)^T + X * X^T;
	//     = X * X^T -(1 / N) * (X_1N) * (X_1N)^T;
	alpha = - 1.0f / static_cast<float>(N);
	Ssyr(&alpha, X_1N, varCovMatrix);
	//varCovMatrixの値は上半分のみ設定される
	
	//varCovMatrix = (1 / N) * varCovMatrix;
	//     = (1 / N) * (X * X^T -(1 / N)*(X_1N) * (X_1N)^T);
	alpha = 1.0f / static_cast<float>(N);
	Sscal(&alpha, varCovMatrix);
	//この時点で分散共分散行列varCovMatrixが取得できた
	//ただしvarCovMatrixの値は上半分のみ設定されている
	
	-------------------------------------------------------------------------- */
	
	//varCovMatrix = E * diag(W) * E^T
	//E : 直交行列(varCovMatrixの固有ベクトルから成る)
	//W : varCovMatrixの固有値から成るベクトル
	//    W = (l_0, l_1, ... , l_(D-1)), l_i <= l_(i+1) i=0, ... ,D-2
	DeviceMatrix E;
	DeviceVector W;
	DnSsyevd(varCovMatrix, W , E);
	varCovEigenVector = E;
	varCovEigenValue = W;
	
	//NULLストリームの完了待ち:Wの算出待ち
	CUDA_CALL(cudaStreamSynchronize(0));
	
	//W = W^(-1/2)
	invSqrtByElement(W);
	
	//ET = E^T;
	//   = 1.0f * E^T + 0.0f * E;
	DeviceMatrix ET(D, D);
	float alpha = 1.0f;
	float beta  = 0.0f;
	Sgeam(&alpha, CUBLAS_OP_T, E, &beta, CUBLAS_OP_N, E, ET);
	
	//P_PCA =     diag(W) * E^T;
	pcaWhiteningMatrix = DeviceMatrix(D, D);
	Sdgmm(W, ET, pcaWhiteningMatrix);
	
	//NULLストリームの完了待ち:
	CUDA_CALL(cudaStreamSynchronize(0));
	
	//W_inv = diag(W)^(-1)
	DeviceVector W_inv;
	W_inv = W;
	invByElement(W_inv);
	
	//(P_PCA)^(-1) = E * diag(W)^(-1)
	inversePCAWhiteningMatrix = DeviceMatrix(D, D);
	Sdgmm(E, W_inv, inversePCAWhiteningMatrix);
	
	//P_ZCA = E * diag(W) * E^T;
	//      = E * P_PCA;
	zcaWhiteningMatrix = DeviceMatrix(D, D);
	alpha = 1.0f;
	beta = 0.0f;
	Sgemm(&alpha, CUBLAS_OP_N, E, CUBLAS_OP_N, pcaWhiteningMatrix, &beta, zcaWhiteningMatrix);
	
	//(P_ZCA)^(-1) = E * diag(W)^(-1) * E^T;
	//             = (P_PCA)^(-1) * E^T;
	inverseZCAWhiteningMatrix = DeviceMatrix(D, D);
	alpha = 1.0f;
	beta = 0.0f;
	Sgemm(&alpha, CUBLAS_OP_N, inversePCAWhiteningMatrix, CUBLAS_OP_N, ET, &beta, inverseZCAWhiteningMatrix);
	
	//NULLストリームの完了待ち
	CUDA_CALL(cudaStreamSynchronize(0));
}

