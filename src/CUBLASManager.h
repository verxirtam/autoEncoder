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


#include <sstream>


#include "cublas_v2.h"

#include "CUDAManager.h"

#include "CuBlasException.h"




#define CUBLAS_CALL(cmd)\
{\
	{\
		cublasStatus_t stat;\
		stat =  cmd;\
		if(stat != CUBLAS_STATUS_SUCCESS)\
		{\
			std::stringstream msg;\
			msg << "CUBLAS_ERROR : ";\
			msg << CUBLASManager::getErrorString(stat) << " at ";\
			msg << __FILE__ << ":";\
			msg << __LINE__ << " ";\
			msg << #cmd << std::endl;\
			throw CuBlasException(msg.str());\
		}\
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

