/*
 * =====================================================================================
 *
 *       Filename:  CuBlasManager.h
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

#include "CudaManager.h"

#include "CuBlasException.h"


namespace cuda
{



#define CUBLAS_CALL(cmd)\
{\
	{\
		cublasStatus_t stat;\
		stat =  cmd;\
		if(stat != CUBLAS_STATUS_SUCCESS)\
		{\
			std::stringstream msg;\
			msg << "CUBLAS_ERROR : ";\
			msg << cuda::CuBlasManager::getErrorString(stat) << " at ";\
			msg << __FILE__ << ":";\
			msg << __LINE__ << " ";\
			msg << #cmd << std::endl;\
			throw cuda::CuBlasException(msg.str());\
		}\
	}\
}




class CuBlasManager
{
private:
	cublasHandle_t handle;
	CuBlasManager():
		handle()
	{
		CUBLAS_CALL(cublasCreate_v2(&handle));
	}
	//シングルトンとするため削除する
	//コピーコンストラクタ
	CuBlasManager(const CuBlasManager&) = delete;
	//コピー代入演算子
	CuBlasManager& operator=(const CuBlasManager&) = delete;
	//ムーブコンストラクタ
	CuBlasManager(CuBlasManager&&) = delete;
	//ムーブ代入演算子
	CuBlasManager& operator=(CuBlasManager&&) = delete;
public:
	virtual ~CuBlasManager()
	{
		CUBLAS_CALL(cublasDestroy(handle));
	}
	static CuBlasManager& getInstance()
	{
		static CuBlasManager i;
		return i;
	}
	static cublasHandle_t getHandle()
	{
		return getInstance().handle;
	}
	static const char* getErrorString(cublasStatus_t error);
};

}

