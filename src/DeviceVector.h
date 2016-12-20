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
	unsigned int dimension;
	float* device;
public:
	//コピーコンストラクタ
	DeviceVector(const DeviceVector& dv):
		dimension(dv.dimension),
		device(nullptr)
	{
		//dimensionが0でなければ
		//デバイスメモリ確保とデバイスメモリ同士でのコピーを行う
		//dimensionが0の時はdv.device = nullptrのためコピーしない
		if(this->dimension != 0)
		{
			cudaMalloc((void**)&(this->device), this->dimension * sizeof(float));
			cudaMemcpy(this->device, dv.device, this->dimension * sizeof(float), cudaMemcpyDeviceToDevice);
		}
	}
	//コピー代入演算子
	DeviceVector& operator=(const DeviceVector& dv)
	{
		//次元のコピー
		this->dimension = dv.dimension;
		
		//コピー先のメモリ解放
		cudaFree(this->device);
		this->device = nullptr;
		
		//dimensionが0でなければ
		//デバイスメモリ確保とデバイスメモリ同士でのコピーを行う
		//dimensionが0の時はdv.device = nullptrのためコピーしない
		if(this->dimension != 0)
		{
			cudaMalloc((void**)&(this->device), this->dimension * sizeof(float));
			cudaMemcpy(this->device, dv.device, this->dimension * sizeof(float), cudaMemcpyDeviceToDevice);
		}
		//自身への参照を返す
		return *this;
	}
	//ムーブコンストラクタ
	//dvのアドレスを付け替える
	DeviceVector(DeviceVector&& dv):
		dimension(dv.dimension),
		device(dv.device)
	{
		//コピー元はnullptrにする
		dv.device = nullptr;
	}
	//ムーブ代入演算子
	//dvのアドレスを付け替える
	DeviceVector& operator=(DeviceVector&& dv)
	{
		//次元のコピー
		this->dimension = dv.dimension;
		
		//コピー先のメモリ解放
		cudaFree(this->device);
		//ムーブするアドレスの付け替え
		this->device = dv.device;
		//ムーブ元のアドレスをnullptrにする
		dv.device = nullptr;
		//自身への参照を返す
		return *this;
	}
	//コンストラクタ
	//(デフォルトコンストラクタにもなっている)
	DeviceVector(unsigned int d = 0):
		dimension(d),
		device(nullptr)
	{
		if(d != 0)
		{
			cudaMalloc((void**)&device, dimension * sizeof(float));
		}
	}
	~DeviceVector()
	{
		cudaFree(device);
	}
	void set(const float* const host)
	{
		cublasStatus_t stat;
		stat = cublasSetVector(dimension, sizeof(float), host, 1, device, 1);
		if(stat != CUBLAS_STATUS_SUCCESS)
		{
			std::cout << "error at cublasSetVector()" << std::endl;
		}
	}
	void get(float* const host)
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


