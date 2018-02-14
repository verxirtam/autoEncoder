/*
 * =====================================================================================
 *
 *       Filename:  CudaStreamContainer.h
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

#include "CudaManager.h"

namespace cuda
{

class CudaStreamContainer
{
private:
	std::vector<cudaStream_t> stream;
public:
	//コンストラクタ
	CudaStreamContainer():
		stream()
	{
	}
	//Streamの初期化
	void init(unsigned int stream_count)
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
	cudaStream_t get(unsigned int stream_index)
	{
		return stream[stream_index];
	}
	unsigned int getCount()
	{
		return stream.size();
	}
};

}

