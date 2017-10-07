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
DeviceMatrix Normalization::getWhitening
	(
		const DeviceMatrix& whiteningMatrix,
		const DeviceMatrix& X,
		const DeviceVector& _1B
	) const
{
	//TODO 使用するストリームを明示する&ストリームの完了待ちを行う
	CUBLAS_CALL(cublasSetStream(CuBlasManager::getHandle(), 0));
	
	unsigned int D = X.getRowCount();
	unsigned int B = X.getColumnCount();
	//auto _1B = DeviceVector::get1Vector(B);//TODO いちいち1Vector作るのは無駄
	//whiteningMatrix * (X - mean * _1B^T)
	
	DeviceMatrix Y = X;
	// Y = (-1.0f) * mean * _1B^T + Y;
	//   = X - mean * _1B^T;
	float alpha = - 1.0f;
	Sger(&alpha, mean, _1B, Y);
	
	// nX = 1.0f * whiteningMatrix * Y + 0.0f * nX;
	//    = whiteningMatrix * Y;
	//    = whiteningMatrix * (X - mean * _1B^T);
	DeviceMatrix nX(D, B);
	alpha = 1.0f;
	float beta = 0.0f;
	Sgemm(&alpha, CUBLAS_OP_N, whiteningMatrix, CUBLAS_OP_N, Y, &beta, nX);
	
	//NULLストリームの完了待ち
	CUDA_CALL(cudaStreamSynchronize(0));
	
	return nX;
}

//白色化の逆変換を行う
DeviceMatrix Normalization::getInverseWhitening
	(
		const DeviceMatrix& inverseWhiteningMatrix,
		const DeviceMatrix& nX,
		const DeviceVector& _1B
	) const
{
	//TODO 使用するストリームを明示する&ストリームの完了待ちを行う
	CUBLAS_CALL(cublasSetStream(CuBlasManager::getHandle(), 0));
	
	unsigned int D = nX.getRowCount();
	unsigned int B = nX.getColumnCount();
	//auto _1B = DeviceVector::get1Vector(B);//TODO いちいち1Vector作るのは無駄
	//                          nX                = whiteningMatrix * (X - mean * _1B^T)
	// inverseWhiteningMatrix * nX                = X - mean * _1B^T
	// inverseWhiteningMatrix * nX + mean * _1B^T = X
	// ->
	// X = inverseWhiteningMatrix * nX + mean * _1B^T
	
	// Y = 1.0f * inverseWhiteningMatrix * nX + 0.0f * Y;
	//   = inverseWhiteningMatrix * nX;
	DeviceMatrix Y(D, B);
	float alpha = 1.0f;
	float  beta = 0.0f;
	Sgemm(&alpha, CUBLAS_OP_N, inverseWhiteningMatrix, CUBLAS_OP_N, nX, &beta, Y);
	
	DeviceMatrix X = Y;
	// X = 1.0f * mean * _1B^T + Y;
	//   = Y + mean * _1B^T;
	//   = inverseWhiteningMatrix * nX + mean * _1B^T;
	alpha = 1.0f;
	Sger(&alpha, mean, _1B, X);
	
	//NULLストリームの完了待ち
	CUDA_CALL(cudaStreamSynchronize(0));
	
	return X;
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

