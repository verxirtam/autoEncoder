
#pragma once

#include "../cuda/DeviceMatrix.h"


namespace nn
{

class LayerInput
{
private:
	using DeviceMatrix = cuda::DeviceMatrix;
	using DeviceVector = cuda::DeviceVector;
	//ミニバッチサイズ
	unsigned int miniBatchSize;
	//インプット
	DeviceMatrix x;
public:
	//コンストラクタ
	LayerInput():
		miniBatchSize(1),
		x()
	{
	}
	//初期化
	void init(unsigned int dim_input, unsigned int minibatch_size = 1)
	{
		miniBatchSize = miniBatchSize;
		x = DeviceMatrix(dim_input, miniBatchSize);
	}
	//順伝播
	const DeviceMatrix& forward(const DeviceMatrix& x_)
	{
		x = x_;
		return x;
	}
	//逆伝播
	const DeviceMatrix& back(const DeviceMatrix& weight_t_delta)
	{
		return weight_t_delta;
	}
	//パラメータの更新
	void update(const DeviceMatrix& x)
	{
	}
	const DeviceMatrix& getZ() const
	{
		return x;
	}
	unsigned int getMiniBatchSize() const
	{
		return miniBatchSize;
	}
};


}//namespace nn

