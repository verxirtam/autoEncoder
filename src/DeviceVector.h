/*
 * =====================================================================================
 *
 *       Filename:  DeviceVector.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月19日 02時36分49秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include "CUBLASManager.h"

class DeviceVector
{
private:
	const int dimension;
	float* device;
	//ポインタメンバを持つがコピー不要なので禁止する
	//本当はデバイスメモリ同士でコピーしたい...。
	//コピーコンストラクタ
	DeviceVector(const DeviceVector&) = delete;
	//コピー代入演算子
	DeviceVector& operator=(const DeviceVector&) = delete;
	//ムーブコンストラクタ
	DeviceVector(DeviceVector&&) = delete;
	//ムーブ代入演算子
	DeviceVector& operator=(DeviceVector&&) = delete;
public:
	DeviceVector(int d):
		dimension(d),
		device(nullptr)
	{
		cudaMalloc((void**)&device, dimension * sizeof(float));
	}
	~DeviceVector()
	{
		cudaFree(device);
	}
	void set(const float* host)
	{
		cublasStatus_t stat;
		stat = cublasSetVector(dimension, sizeof(float), host, 1, device, 1);
		if(stat != CUBLAS_STATUS_SUCCESS)
		{
			std::cout << "error at cublasSetVector()" << std::endl;
		}
	}
	void get(float* host)
	{
		cublasStatus_t stat;
		stat = cublasGetVector(dimension, sizeof(float), device, 1, host, 1);
		if(stat != CUBLAS_STATUS_SUCCESS)
		{
			std::cout << "error at cublasGetVector()" << std::endl;
		}
	}
	float* getAddress() const
	{
		return device;
	}
	int getDimension() const
	{
		return dimension;
	}
};


