/*
 * =====================================================================================
 *
 *       Filename:  CUBLASManager.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月19日 02時35分00秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include "cublas_v2.h"


#define CUDA_CALL(cmd)\
{\
	cudaError_t error;\
	error =  cmd;\
	if(error != cudaSuccess)\
	{\
		std::cout << "CUDA_ERROR : ";\
		std::cout << cudaGetErrorString(error) << " at ";\
		std::cout << __FILE__ << " ";\
		std::cout << __LINE__ << " ";\
		std::cout << #cmd << std::endl;\
	}\
}



#define CUBLAS_CALL(cmd)\
{\
	cublasStatus_t stat;\
	stat =  cmd;\
	if(stat != CUBLAS_STATUS_SUCCESS)\
	{\
		std::cout << "CUBLAS_ERROR : ";\
		std::cout << CUBLASManager::getErrorString(stat) << " at ";\
		std::cout << __FILE__ << " ";\
		std::cout << __LINE__ << " ";\
		std::cout << #cmd << std::endl;\
	}\
}




class CUBLASManager
{
private:
	cublasHandle_t handle;
	CUBLASManager():
		handle()
	{
		CUBLAS_CALL(cublasCreate_v2(&handle));
	}
	//シングルトンとするため削除する
	//コピーコンストラクタ
	CUBLASManager(const CUBLASManager&) = delete;
	//コピー代入演算子
	CUBLASManager& operator=(const CUBLASManager&) = delete;
	//ムーブコンストラクタ
	CUBLASManager(CUBLASManager&&) = delete;
	//ムーブ代入演算子
	CUBLASManager& operator=(CUBLASManager&&) = delete;
public:
	virtual ~CUBLASManager()
	{
		CUBLAS_CALL(cublasDestroy(handle));
	}
	static CUBLASManager& getInstance()
	{
		static CUBLASManager i;
		return i;
	}
	static cublasHandle_t getHandle()
	{
		return getInstance().handle;
	}
	static const char* getErrorString(cublasStatus_t error);
};

