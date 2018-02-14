
#pragma once

#include "../cuda/DeviceMatrix.h"

namespace nn
{

class LayerNull
{
private:
	using DeviceMatrix = cuda::DeviceMatrix;
	using DeviceVector = cuda::DeviceVector;
	DeviceMatrix z;
public:
	//コンストラクタ
	LayerNull():
		z()
	{
	}
	//初期化
	void init()
	{
	}
	//順伝播
	const DeviceMatrix& forward(const DeviceMatrix& x)
	{
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
	//zを取得
	const DeviceMatrix& getZ() const
	{
		return z;
	}
};



}



