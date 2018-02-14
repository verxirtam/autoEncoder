/*
 * =====================================================================================
 *
 *       Filename:  UpdateMethodMomentum.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年02月05日 01時43分04秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include "../cuda/DeviceMatrix.h"
#include "../cuda/DeviceVector.h"

#include "../cuda/CuBlasFunction.h"

namespace nn
{

class UpdateMethodMomentum
{
private:
	using DeviceMatrix = cuda::DeviceMatrix;
	using DeviceVector = cuda::DeviceVector;
	//weightの更新差分
	DeviceMatrix deltaWeight;
	//biasの更新差分
	DeviceVector deltaBias;
	//(minibatch_size)次元の1-vector
	DeviceVector _1B;
	//学習係数
	float learningRate;
	//モメンタム
	float momentum;
public:
	//コンストラクタ
	UpdateMethodMomentum():
		deltaWeight(),
		deltaBias(),
		_1B(),
		learningRate(0.125f),
		momentum(0.875f)
	{
	}
	//初期化
	void init(unsigned int dim_input, unsigned int dim_output, unsigned int minibatch_size);
	//パラメータの更新
	void update(const DeviceMatrix& x, const DeviceMatrix& delta, DeviceMatrix& weight, DeviceVector& bias);
	//setter
	void setLearningRate(float learning_rate)
	{
		learningRate = learning_rate;
	}
	void setMomentum(float momentum_)
	{
		momentum = momentum_;
	}
	void setDeltaWeight(const DeviceMatrix& delta_weight)
	{
		deltaWeight = delta_weight;
	}
	void setDeltaBias(const DeviceVector& delta_bias)
	{
		deltaBias = delta_bias;
	}
	//getter
	const DeviceMatrix& getDeltaWeight() const
	{
		return deltaWeight;
	}
	const DeviceVector& getDeltaBias() const
	{
		return deltaBias;
	}
	float getLearningRate() const
	{
		return learningRate;
	}
	float getMomentum() const
	{
		return momentum;
	}
};


}


