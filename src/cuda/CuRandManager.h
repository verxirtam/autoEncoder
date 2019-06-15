/*
 * =====================================================================================
 *
 *       Filename:  CuRandManager.h
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

#include <sstream>
#include <vector>

#include <ctime>

#include <curand.h>

#include "CuRandException.h"

namespace cuda
{

#define CURAND_CALL(cmd)\
{\
	{\
		curandStatus_t error;\
		error =  cmd;\
		if(error != CURAND_STATUS_SUCCESS)\
		{\
			std::stringstream msg;\
			msg << "CURAND_ERROR : ";\
			msg << cuda::CuRandManager::getErrorString(error) << " at ";\
			msg << __FILE__ << ":";\
			msg << __LINE__ << " ";\
			msg << __PRETTY_FUNCTION__ << " ";\
			msg << #cmd << std::endl;\
			throw cuda::CuRandException(msg.str());\
		}\
	}\
}
#define CURAND_CALL_NO_EXCEPTION(cmd)\
{\
	{\
		cmd;\
	}\
}

//cuRAND全体の管理を行う
class CuRandManager
{
private:
	curandGenerator_t generator;
	CuRandManager():
		generator()
	{
		//擬似乱数generatorの生成
		CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937));
		//seedの設定
		//現在時刻を元にseedを設定
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, static_cast<unsigned long long>(std::time(nullptr))));
	}
	~CuRandManager()
	{
		CURAND_CALL_NO_EXCEPTION(curandDestroyGenerator(generator));
	}
	//シングルトンとするため禁止する
	//コピーコンストラクタ
	CuRandManager(const CuRandManager&) = delete;
	//コピー代入演算子
	CuRandManager& operator=(const CuRandManager&) = delete;
	//ムーブコンストラクタ
	CuRandManager(CuRandManager&&) = delete;
	//ムーブ代入演算子
	CuRandManager& operator=(CuRandManager&&) = delete;
public:
	static CuRandManager& getInstance()
	{
		static CuRandManager crm;
		return crm;
	}
	static curandGenerator_t getGenerator()
	{
		return CuRandManager::getInstance().generator;
	}
	static const char* getErrorString(curandStatus_t error);
};

}


