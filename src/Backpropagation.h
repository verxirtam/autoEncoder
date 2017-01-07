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

#include "BackpropagationException.h"


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
public:
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
	void obtainDEDW(unsigned int l, unsigned int thread_count = 256, unsigned int d = 64);
	//dEdW[l] = delta[l] * (z[l -1])^T;
	template<unsigned int D>
	void obtainDEDWMain(unsigned int l, unsigned int thread_count);
	
	//コンストラクタ
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
		this->setSubStreamCount(4);
	}
	//初期化
	void init(const std::vector<unsigned int>& unit_count);
	//weightをランダムに初期化する
	void initRandom(void);
	//順伝播
	void forward(const std::vector<float>& x, std::vector<float>& y);
	//逆伝播
	void back(const std::vector<float>& d);
	
	void updateParameter();

	void setWeight(const std::vector<std::vector<float> >& w)
	{
		if(w.size() != this->weight.size())
		{
			throw BackpropagationException("error at setWeight() :  w.size() != this->weight.size().");
		}
		unsigned int imax = this->weight.size();
		for(unsigned int i = 0; i < imax; i++)
		{
			this->weight[i].set(w[i]);
		}
	}
	
	void setBias(const std::vector<std::vector<float> >& b)
	{
		if(b.size() != this->bias.size())
		{
			throw BackpropagationException("error at setBias() :  b.size() != this->bias.size().");
		}
		unsigned int imax = this->weight.size();
		for(unsigned int i = 0; i < imax; i++)
		{
			this->bias[i].set(b[i]);
		}
		
	}
	
	
	const std::vector<DeviceVector>& getU() const
	{
		return this->u;
	}
	void getU(std::vector<std::vector<float> >& hu) const
	{
		std::vector<float> h;
		unsigned int imax = this->u.size();
		for(unsigned int i = 0; i < imax; i++)
		{
			this->u[i].get(h);
			hu.push_back(h);
		}
		
	}
	std::vector<std::vector<float> > getUAsVector() const
	{
		std::vector<std::vector<float> > hu;
		for(auto&& _u : this->u )
		{
			hu.push_back(_u.get());
		}
		return hu;
	}
	const std::vector<DeviceVector>& getZ() const
	{
		return this->z;
	}
	std::vector<std::vector<float> > getZAsVector() const
	{
		std::vector<std::vector<float> > hz;
		for(auto&& _z : this->z )
		{
			hz.push_back(_z.get());
		}
		return hz;
	}
	const std::vector<DeviceMatrix>& getWeight() const
	{
		return this->weight;
	}
	std::vector<std::vector<float> > getWeightAsVector() const
	{
		std::vector<std::vector<float> > hweight;
		for(auto&& _w : this->weight)
		{
			hweight.push_back(_w.get());
		}
		return hweight;
	}
	const std::vector<DeviceVector>& getBias() const
	{
		return this->bias;
	}
	std::vector<std::vector<float> > getBiasAsVector() const
	{
		std::vector<std::vector<float> > hbias;
		for(auto&& _b : this->bias)
		{
			hbias.push_back(_b.get());
		}
		return hbias;
	}
	const std::vector<DeviceMatrix>& getDEDW() const
	{
		return this->dEdW;
	}
	std::vector<std::vector<float> > getDEDWAsVector() const
	{
		std::vector<std::vector<float> > hdEdW;
		for(auto&& _d : this->dEdW)
		{
			hdEdW.push_back(_d.get());
		}
		return hdEdW;
	}
	const std::vector<DeviceVector>& getDEDB() const
	{
		return this->dEdb;
	}
	const std::vector<DeviceVector>& getDelta() const
	{
		return this->dEdb;
	}
	std::vector<std::vector<float> > getDeltaAsVector() const
	{
		std::vector<std::vector<float> > hdelta;
		for(auto&& _d : this->delta)
		{
			hdelta.push_back(_d.get());
		}
		return hdelta;
	}
	std::vector<std::vector<float> > getDEDBAsVector() const
	{
		return getDeltaAsVector();
	}
	const std::vector<DeviceVector>& getWTDelta() const
	{
		return this->WTdelta;
	}
	void setSubStreamCount(unsigned int substream_count) const
	{
		CUDAManager::getInstance().initStream(substream_count + 1);
	}
	unsigned int getSubStreamCount(void) const
	{
		return CUDAManager::getInstance().getStreamCount() - 1;
	}
	cudaStream_t getMainStream(void) const
	{
		return CUDAManager::getInstance().getStream(0);
	}
	cudaStream_t getSubStream(unsigned int stream_index) const
	{
		return CUDAManager::getInstance().getStream(stream_index + 1);
	}
};


