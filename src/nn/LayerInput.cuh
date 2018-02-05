
#pragma once

#include "../cuda/DeviceMatrix.h"


namespace nn
{

class LayerInput
{
private:
	using DeviceMatrix = cuda::DeviceMatrix;
	using DeviceVector = cuda::DeviceVector;
	//インプット
	DeviceMatrix x;
public:
	//コンストラクタ
	LayerInput():
		x()
	{
	}
	//初期化
	void init(unsigned int dim_input, unsigned int minibatch_size = 1)
	{
		x = DeviceMatrix(dim_input, minibatch_size);
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
};


}//namespace nn

