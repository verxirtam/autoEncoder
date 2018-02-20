
#pragma once

#include "../cuda/DeviceMatrix.h"
#include "../cuda/DeviceVector.h"

namespace nn
{

template <class ActivateMethod, class UpdateMethod>
class Layer
{
private:
	using DeviceMatrix = cuda::DeviceMatrix;
	using DeviceVector = cuda::DeviceVector;
	//ミニバッチサイズ
	unsigned int miniBatchSize;
	//ノード間の重み
	DeviceMatrix weight;
	//ノード間のバイアス
	DeviceVector bias;
	//u = weight * z_0 + bias
	//z_0は入力
	DeviceMatrix u;
	//z = ActivateMethod(u)
	DeviceMatrix z;
	//miniBatchSize次元の1-vector
	DeviceVector _1B;
	//delta = dE/du = (dE/du_ij)_ij
	DeviceMatrix delta;
	//weightTDelta = weight ^ T * delta
	DeviceMatrix weightTDelta;
	//パラメータ更新の手法
	UpdateMethod updateMethod;
	
	//weight, biasをランダムに初期化する
	void initWeightBias(void);
	//順伝播の線型部分
	//u = weight * x + bias * _1B ^ T
	void forwardLinear(const DeviceMatrix& x);
	
	//weightTDeltaを算出する
	//weightTDelta = weight^T * delta;
	void getWeightTDelta();

public:
	//コンストラクタ
	Layer():
		miniBatchSize(1),
		weight(),
		bias(),
		u(),
		z(),
		_1B(),
		delta(),
		updateMethod()
	{
	}
	//初期化
	void init(unsigned int dim_input, unsigned int dim_output, unsigned int minibatch_size = 1);
	//順伝播
	const DeviceMatrix& forward(const DeviceMatrix& x);
	//逆伝播
	const DeviceMatrix& back(const DeviceMatrix& delta_output);
	//パラメータの更新
	void update(const DeviceMatrix& x);
	
	//setter
	void setWeight(const DeviceMatrix& w)
	{
		weight = w;
	}
	void setBias(const DeviceVector& b)
	{
		bias = b;
	}
	
	//getter
	unsigned int getMiniBatchSize() const
	{
		return miniBatchSize;
	}
	const DeviceMatrix& getWeight() const
	{
		return weight;
	}
	const DeviceVector& getBias() const
	{
		return bias;
	}
	const DeviceMatrix& getU() const
	{
		return u;
	}
	const DeviceMatrix& getZ() const
	{
		return z;
	}
	const DeviceMatrix& getDelta() const
	{
		return delta;
	}
	const DeviceMatrix& getWeightTDelta() const
	{
		return weightTDelta;
	}
	UpdateMethod& getUpdateMethod()
	{
		return updateMethod;
	}
};



}




#include "Layer_detail.h"


