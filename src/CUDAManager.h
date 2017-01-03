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

#include <vector>

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
	std::vector<cudaStream_t> stream;
	CUDAManager():
		deviceProp(),
		stream()
	{
		cudaGetDeviceProperties(&deviceProp,0);
	}
	~CUDAManager()
	{
		for(auto&& s : stream)
		{
			CUDA_CALL(cudaStreamDestroy(s));
		}
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
	static CUDAManager& getInstance()
	{
		static CUDAManager cm;
		return cm;
	}
	static inline const cudaDeviceProp& getDeviceProp(void)
	{
		return CUDAManager::getInstance().deviceProp;
	}
	//Streamの初期化
	void initStream(unsigned int stream_count)
	{
		//既存のStreamの破棄
		for(auto&& s : stream)
		{
			CUDA_CALL(cudaStreamDestroy(s));
		}
		stream.clear();
		//Streamの新規作成
		stream = std::vector<cudaStream_t>(stream_count);
		//Streamの生成
		for(auto&& s : stream)
		{
			CUDA_CALL(cudaStreamCreate(&s));
		}
	}
	cudaStream_t getStream(unsigned int stream_index)
	{
		return stream[stream_index];
	}
	unsigned int getStreamCount()
	{
		return stream.size();
	}
};


