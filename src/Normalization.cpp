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


//正規化用のデータを元に平均・白色化の変換行列を作成する
void Normalization::init(const DeviceMatrix& X)
{
	//データの次元
	unsigned int D = X.getRowCount();
	//データ数
	unsigned int N = X.getColumnCount();
	//全成分が1のベクトル
	auto _1N = DeviceVector::get1Vector(N);
	//X * _1N (* は内積)
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
	
	//varCovMatrix = E * diag(W) * E^T
	//E : 直交行列
	//W : varCovMatrixの固有値から成るベクトル
	//    W = (l_0, l_1, ... , l_(D-1)), l_i <= l_(i+1) i=0, ... ,D-2
	DeviceMatrix E;
	DeviceVector W;
	DnSsyevd(varCovMatrix, W , E);
	
	//W = W^(-1/2)
	invSqrtByElement(W);
	
	//ET = E^T;
	//   = 1.0f * E^T + 0.0f * E;
	DeviceMatrix ET(D, D);
	alpha = 1.0f;
	beta = 0.0f;
	Sgeam(&alpha, CUBLAS_OP_T, E, &beta, CUBLAS_OP_N, E, ET);
	
	//P_PCA =     diag(W) * E^T;
	pcaWhiteningMatrix = DeviceMatrix(D, D);
	Sdgmm(W, ET, pcaWhiteningMatrix);
	
	
	//P_ZCA = E * diag(W) * E^T;
	//      = E * P_PCA;
	zcaWhiteningMatrix = DeviceMatrix(D, D);
	alpha = 1.0f;
	beta = 0.0f;
	Sgemm(&alpha, CUBLAS_OP_N, E, CUBLAS_OP_N, pcaWhiteningMatrix, &beta, zcaWhiteningMatrix);

}

