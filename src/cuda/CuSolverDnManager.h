/*
 * =====================================================================================
 *
 *       Filename:  CuSolverDnManager.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年01月22日 22時13分06秒
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

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "CuSolverDnException.h"

namespace cuda
{

#define CUSOLVERDN_CALL(cmd)\
{\
	{\
		cusolverStatus_t status;\
		status =  cmd;\
		if(status != CUSOLVER_STATUS_SUCCESS)\
		{\
			std::stringstream msg;\
			msg << "CUSOLVERDN_ERROR : ";\
			msg << CuSolverDnManager::getErrorString(status) << " at ";\
			msg << __FILE__ << ":";\
			msg << __LINE__ << " ";\
			msg << __PRETTY_FUNCTION__ << " ";\
			msg << #cmd << std::endl;\
			throw CuSolverDnException(msg.str());\
		}\
	}\
}

//CuSolverDnの設定・変更・情報取得を行う
//シングルトンとしている。
class CuSolverDnManager
{
private:
	cusolverDnHandle_t handle;
	
	CuSolverDnManager():
		handle(nullptr)
	{
		CUSOLVERDN_CALL(cusolverDnCreate(&handle));
	}
	~CuSolverDnManager()
	{
		CUSOLVERDN_CALL(cusolverDnDestroy(handle));
	}
	//シングルトンとするため禁止する
	//コピーコンストラクタ
	CuSolverDnManager(const CuSolverDnManager&) = delete;
	//コピー代入演算子
	CuSolverDnManager& operator=(const CuSolverDnManager&) = delete;
	//ムーブコンストラクタ
	CuSolverDnManager(CuSolverDnManager&&) = delete;
	//ムーブ代入演算子
	CuSolverDnManager& operator=(CuSolverDnManager&&) = delete;
public:
	static inline CuSolverDnManager& getInstance()
	{
		static CuSolverDnManager cm;
		return cm;
	}
	static inline cusolverDnHandle_t getHandle(void)
	{
		return CuSolverDnManager::getInstance().handle;
	}
	static const char* getErrorString(cusolverStatus_t status);
};

}

