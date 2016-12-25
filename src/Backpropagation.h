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
	
	const unsigned int layerCount;
	std::vector<unsigned int> unitCount;
	float epsilon;
	std::vector<DeviceVector> u;
	std::vector<DeviceVector> z;
	std::vector<DeviceMatrix> weight;
	std::vector<DeviceVector> bias;
	std::vector<DeviceMatrix> dEdW;
	std::vector<DeviceVector> dEdb;
	std::vector<DeviceVector>& delta;
	std::vector<DeviceVector> WTdelta;
	//活性化関数
	float f(const float& x)
	{
		return std::tanh(x);
	}
	//活性化関数の微分
	float df(const float& x)
	{
		float tanh_x = std::tanh(x);
		return 1.0f - (tanh_x * tanh_x);
	}
	//下記を求める
	//u[l + 1] = weight[l + 1] * z[l] + bias[l + 1];
	void obtainUFromZ(unsigned int l)
	{
		//Sgemv()を使用するため事前に下記を算出する
		u[l + 1] = bias[l + 1];
		
		//Sgemv()を用いて下記を求める
		//u[l + 1] = weight[l + 1] * z[l] + u[l + 1];
		float alpha = 1.0f;
		float beta = 1.0f;
		Sgemv(&alpha, CUBLAS_OP_N, weight[l + 1], z[l], &beta, u[l + 1]);
	}
	//下記を求める
	//z[l + 1] = f(u[l + 1]);
	void obtainZFromU(unsigned int l);
	
	//逆伝播の初期化
	//delta[layer_count - 1] = u[layer_count - 1] - dd;
	void obtainDeltaLast(const std::vector<float>& d);
	//逆伝播でのdeltaの算出
	//lについて降順に逐次実行("**"は要素ごとの積(cudaで実行))
	//delta[l] = f'(u[l]) ** ((W[l + 1])^T * delta[l+1]);
	//l = layerCount - 2, ... , 1
	void obtainDelta(unsigned int l);
	//delta[l] = f'(u[l]) ** WTdelta[l + 1];
	void obtainDeltaFromFdUWTDelta(unsigned int l);
	//dEdW[l] = delta[l] * (z[l -1])^T;
	void obtainDEDW(unsigned int l);
public:
	Backpropagation(unsigned int layer_count):
		layerCount(layer_count),
		unitCount(),
		epsilon(0.0625f * 0.0625f),
		u(),
		z(),
		weight(),
		bias(),
		dEdW(),
		dEdb(),
		delta(dEdb),
		WTdelta()
	{
	}
	void init(const std::vector<unsigned int>& unit_count);
	void initRandom(void);
	
	void forward(const std::vector<float>& x, std::vector<float>& y)
	{
		z[0].set(x);
		for(unsigned int l = 0; l < layerCount - 1; l++)
		{
			//z[l], weight[l+1], bias[l+1]からu[l+1]を得る
			obtainUFromZ(l);
			//u[l+1]からz[l+1]を得る
			obtainZFromU(l);
		}
		//z[L-1] == yを設定
		z[layerCount - 1].get(y);
	}
	//逆伝播
	void back(const std::vector<float>& d);
	
	void updateParameter();
};


