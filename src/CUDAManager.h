/*
 * =====================================================================================
 *
 *       Filename:  CUDAManager.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月31日 06時40分56秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#pragma once


#include <cuda_runtime.h>

#include "CudaException.h"

#define CUDA_CALL(cmd)\
{\
	{\
		cudaError_t error;\
		error =  cmd;\
		if(error != cudaSuccess)\
		{\
			std::stringstream msg;\
			msg << "CUDA_ERROR : ";\
			msg << cudaGetErrorString(error) << " at ";\
			msg << __FILE__ << ":";\
			msg << __LINE__ << " ";\
			msg << __PRETTY_FUNCTION__ << " ";\
			msg << #cmd << std::endl;\
			throw CudaException(msg.str());\
		}\
	}\
}

//CUDA全体の設定・変更・情報取得を行う
//シングルトンとしている。
//コンストラクタでデバイスの情報を取得している。
//前提：
//	CUDAデバイスは1台のみ
class CUDAManager
{
private:
	cudaDeviceProp deviceProp;
	static CUDAManager& getInstance()
	{
		static CUDAManager cm;
		return cm;
	}
	CUDAManager():
		deviceProp()
	{
		cudaGetDeviceProperties(&deviceProp,0);
	}
	//シングルトンとするため禁止する
	//コピーコンストラクタ
	CUDAManager(const CUDAManager&) = delete;
	//コピー代入演算子
	CUDAManager& operator=(const CUDAManager&) = delete;
	//ムーブコンストラクタ
	CUDAManager(CUDAManager&&) = delete;
	//ムーブ代入演算子
	CUDAManager& operator=(CUDAManager&&) = delete;
public:
	static inline const cudaDeviceProp& getDeviceProp(void)
	{
		//プログラム実行中変化しない値なのでstaticにしてしまう
		return CUDAManager::getInstance().deviceProp;
	}
};


