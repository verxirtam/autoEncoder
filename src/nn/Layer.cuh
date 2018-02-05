
#pragma once

#include "../cuda/DeviceMatrix.h"

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
	//zを取得
	const DeviceMatrix& getZ() const
	{
		return z;
	}
};



}




#include "Layer_detail.cuh"


