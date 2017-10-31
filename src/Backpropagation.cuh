/*
 * =====================================================================================
 *
 *       Filename:  Backpropagation.cuh
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

#include "CuBlasFunction.h"
#include "CuRandManager.h"
#include "BackpropagationException.h"


#include "ActivateFunction.cuh"
#include "Func2to1ElementWiseProduct.cuh"
#include "ElementWiseFunction2to1.cuh"
#include "ElementWiseFunctionUtil.cuh"

template<class AF, class OutputLayer>
class Backpropagation_Base
{
private:
	
	const unsigned int layerCount;
	unsigned int miniBatchSize;
	std::vector<unsigned int> unitCount;
	float epsilon;
	float gamma;
	std::vector<DeviceMatrix> u;
	std::vector<DeviceMatrix> z;
	std::vector<DeviceMatrix> weight;
	std::vector<DeviceVector> bias;
	std::vector<DeviceMatrix> delta;
	std::vector<DeviceMatrix> WTdelta;
	std::vector<DeviceMatrix> deltaWeight;
	std::vector<DeviceVector> deltaBias;
	DeviceVector _1B;
	//weightとbiasをランダムに初期化する
	void initRandom(void);
public:
	
	
	//下記を求める
	//u[l + 1] = weight[l + 1] * z[l] + bias[l + 1] * _1B ^ T;
	void obtainUFromZ(unsigned int l)
	{
		//u[l + 1] = weight[l + 1] * z[l];
		//         = 1.0f * weight[l + 1] * z[l] + 0.0f * u[l + 1];
		float alpha = 1.0f;
		float beta  = 0.0f;
		Sgemm(&alpha, CUBLAS_OP_N, weight[l + 1], CUBLAS_OP_N, z[l], &beta, u[l + 1]);
		
		//u[l + 1] = 1.0f * bias[l + 1] * _1B ^ T + u[l + 1];
		//<=>
		//u[l + 1] = weight[l + 1] * z[l] + bias[l + 1] * _1B ^ T;
		alpha = 1.0;
		Sger(&alpha, bias[l +1], _1B, u[l + 1]);
		
		/////////////////////////////////////////////////////////
		
		//Sgemv()を使用するため事前に下記を算出する
		//u[l + 1] = bias[l + 1];
		
		//Sgemv()を用いて下記を求める
		//u[l + 1] = weight[l + 1] * z[l] + u[l + 1];
		//float alpha = 1.0f;
		//float beta = 1.0f;
		//Sgemv(&alpha, CUBLAS_OP_N, weight[l + 1], z[l], &beta, u[l + 1]);
	}
	//下記を求める
	//z[l + 1] = f(u[l + 1]);
	void obtainZFromU(unsigned int l);
	
	//逆伝播の初期化
	//delta[layer_count - 1] = u[layer_count - 1] - dd;
	void obtainDeltaLast(const DeviceMatrix& d);
	//逆伝播でのdeltaの算出
	//lについて降順に逐次実行("**"は要素ごとの積(cudaで実行))
	//delta[l] = f'(u[l]) ** ((W[l + 1])^T * delta[l+1]);
	//l = layerCount - 2, ... , 1
	void obtainDelta(unsigned int l);
	//delta[l] = f'(u[l]) ** WTdelta[l + 1];
	void obtainDeltaFromFdUWTDelta(unsigned int l);
	
	//コンストラクタ
	Backpropagation_Base(unsigned int layer_count):
		layerCount(layer_count),
		miniBatchSize(1),
		unitCount(),
		epsilon(0.125f),//0.0625f * 0.0625f),
		gamma(0.875f),
		u(),
		z(),
		weight(),
		bias(),
		delta(),
		WTdelta(),
		deltaWeight(),
		deltaBias(),
		_1B()
	{
		this->setSubStreamCount(4);
	}
	//初期化
	void init(const std::vector<unsigned int>& unit_count, unsigned int minibatch_size = 1);
	//順伝播
	void forward(const std::vector<float>& x, std::vector<float>& y);
	void forward(const DeviceMatrix& X, DeviceMatrix& Y);
	//逆伝播
	void back(const std::vector<float>& d);
	void back(const DeviceMatrix& D);
	
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
	
	
	void setEpsilon(float e)
	{
		this->epsilon = (e > 0.0f) ? (e) : (1.0e-6f);
	}
	float getEpsilon() const
	{
		return this->epsilon;
	}
	void setGamma(float g)
	{
		this->gamma = (g > 0.0f) ? (g) : (1.0e-6f);
	}
	float getGamma() const
	{
		return this->gamma;
	}
	
	const std::vector<DeviceMatrix>& getU() const
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
	const std::vector<DeviceMatrix>& getZ() const
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
	const std::vector<DeviceMatrix>& getDelta() const
	{
		return this->delta;
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
	const std::vector<DeviceMatrix>& getWTDelta() const
	{
		return this->WTdelta;
	}
	void setSubStreamCount(unsigned int substream_count) const
	{
		CudaManager::getInstance().initStream(substream_count + 1);
	}
	unsigned int getSubStreamCount(void) const
	{
		return CudaManager::getInstance().getStreamCount() - 1;
	}
	cudaStream_t getMainStream(void) const
	{
		return CudaManager::getInstance().getStream(0);
	}
	cudaStream_t getSubStream(unsigned int stream_index) const
	{
		return CudaManager::getInstance().getStream(stream_index + 1);
	}
};

#include "Backpropagation_detail.h"

#include "Func1to1Tanh.cuh"
#include "OutputLayerRegression.cuh"

using Backpropagation = Backpropagation_Base<Func1to1Tanh, OutputLayerRegression>;

