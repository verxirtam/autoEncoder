/*
 * =====================================================================================
 *
 *       Filename:  CuSolverDnFunction.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年01月23日 00時38分54秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "CuSolverDnFunction.h"


//対称行列に対する固有値・固有ベクトルを求める
// A * V = V * diag(W)
//A 対称行列
//W Aの固有値からなるベクトル(成分は固有値の昇順)
//V Aの固有ベクトル
void DnSsyevd
	(
		const DeviceMatrix& A,
		DeviceVector& W,
		DeviceMatrix& V
	)
{
	//A列数の次元のベクトルを確保
	W = DeviceVector(A.getColumnCount());
	
	//AをVにコピーしてVをAの代わりに使用する
	//Vには最終的に固有ベクトルが格納される
	V = A;
	
	//固有ベクトルを同時に求める
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	//対称行列Aの上半分を算出に使用する
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
	
	
	//ワーク用のバッファサイズ
	int lwork = 0;
	
	//ワーク用のバッファのサイズを取得
	CUSOLVERDN_CALL
		(
			cusolverDnSsyevd_bufferSize
				(
					CuSolverDnManager::getHandle(),
					jobz,
					uplo,
					V.getColumnCount(),
					V.getAddress(),
					V.getColumnCount(),
					W.getAddress(),
					&lwork
				)
		);
	
	//ワーク用のバッファ
	//DeviceVectorとして作成する
	//(例外発生時に開放させるため)
	DeviceVector d_work(lwork);
	
	//結果の成否
	//TODO デバイスメモリdevinfoを保持するクラスを作る
	//TODO 例外発生時に開放できるようにするため
	int* devinfo;
	//デバイスメモリの確保
	CUDA_CALL(cudaMalloc((void**)&devinfo, sizeof(int)));
	//対応するホストメモリ
	int info_gpu = 0;
	
	//固有値・固有ベクトルを算出
	//Wに固有値、
	//Vに固有ベクトルが格納される
	CUSOLVERDN_CALL
		(
			cusolverDnSsyevd
				(
					CuSolverDnManager::getHandle(),
					jobz,
					uplo,
					V.getColumnCount(),
					V.getAddress(),
					V.getColumnCount(),
					W.getAddress(),
					d_work.getAddress(),
					lwork,
					devinfo
				)
		);
	
	//devinfoをホストメモリinfo_gpuにコピー
	CUDA_CALL(cudaMemcpy(&info_gpu, devinfo, sizeof(int), cudaMemcpyDeviceToHost));
	
	//info_gpuが不正な場合は例外発生
	if(info_gpu != 0)
	{
		std::stringstream msg;
		msg << "error at DnSsyevd() : ";
		msg << "info_gpu = " << info_gpu;
		msg << ", ";
		msg << "A = {";
		for(auto&& x : A.get()){msg << x << ", ";}
		msg << "}";
		msg << ", ";
		msg << "V = {";
		for(auto&& x : V.get()){msg << x << ", ";}
		msg << "}";
		msg << ", ";
		msg << "W = {";
		for(auto&& x : W.get()){msg << x << ", ";}
		msg << "}";
		//デバイスメモリの開放
		CUDA_CALL(cudaFree(devinfo));
		throw CuSolverDnException(msg.str());
	}
	
	//デバイスメモリの開放
	CUDA_CALL(cudaFree(devinfo));
	
}

