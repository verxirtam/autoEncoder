/*
 * =====================================================================================
 *
 *       Filename:  Backpropagation.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月19日 02時53分56秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include <cmath>
#include <vector>
#include <memory>
#include <random>

#include "CUBLASFunction.h"

class Backpropagation
{
private:
	using upDeviceVector = std::unique_ptr<DeviceVector>;
	using upDeviceMatrix = std::unique_ptr<DeviceMatrix>;
	
	const unsigned int layerCount;
	std::vector<unsigned int> unitCount;
	std::vector<upDeviceVector> u;
	std::vector<upDeviceVector> z;
	std::vector<upDeviceMatrix> weight;
	std::vector<upDeviceVector> bias;
	std::vector<upDeviceMatrix> dEdW;
	std::vector<upDeviceVector> dEdb;
	float f(const float& x)
	{
		return std::tanh(x);
	}
	float df(const float& x)
	{
		float tanh_x = std::tanh(x);
		return 1.0f - (tanh_x * tanh_x);
	}
	void obtainUFromZ(unsigned int l)
	{
		float alpha = 1.0f;
		float beta = 1.0f;
		cublasScopy(CUBLASManager::getHandle(), bias[l+1]->getDimension(), bias[l+1]->getAddress(), 1, u[l + 1]->getAddress(), 1);
		Sgemv(&alpha, CUBLAS_OP_N, *(weight[l + 1]), *(z[l]), &beta, *(u[l + 1]));
	}
	void obtainZFromU(unsigned int l)
	{
		//TODO CUDAカーネルで実装する
		//下記を全てのjについて実行する<F3>
		//z[l + 1][j] = f(u[l + 1][j])
		//kernel<<<u.dimension, 1>>>(u[l+1]->getAddress(), z[l+1]->getAddress());
	}
public:
	Backpropagation(unsigned int layer_count):
		layerCount(layer_count),
		unitCount(),
		u(),
		z(),
		weight(),
		bias(),
		dEdW(),
		dEdb()
	{
		
	}
	void init(const std::vector<unsigned int>& unit_count);
	void initRandom(void);
	
	void forward(const std::vector<float>& x)
	{
		z[0]->set(x.data());
		for(unsigned int l = 0; l < layerCount - 1; l++)
		{
			//z[l], weight[l+1], bias[l+1]からu[l+1]を得る
			obtainUFromZ(l);
			//u[l+1]からz[l+1]を得る
			obtainZFromU(l);
		}
		//z[L-1] == yまで取得した
	}
};


