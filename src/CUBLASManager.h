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

class CUBLASManager
{
private:
	cublasHandle_t handle;
	CUBLASManager():
		handle()
	{
		cublasStatus_t stat;
		stat = cublasCreate_v2(&handle);
		if(stat != CUBLAS_STATUS_SUCCESS)
		{
			std::cout << "CUBLAS initialization failed" << std::endl;
		}
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
		cublasDestroy(handle);
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

