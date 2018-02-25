#pragma once

#include "../cuda/DeviceMatrix.h"

namespace nn
{

template <class Input, class Internal, class Output>
class Perceptron
{
private:
	using DeviceMatrix = cuda::DeviceMatrix;
	Input input;
	Internal internal;
	Output output;
public:
	Perceptron():
		input(),
		internal(),
		output()
	{
	}
	Input& getInput(void)
	{
		return input;
	}
	const Input& getInput(void) const
	{
		return input;
	}
	Internal& getInternal(void)
	{
		return internal;
	}
	const Internal& getInternal(void) const
	{
		return internal;
	}
	Output& getOutput(void)
	{
		return output;
	}
	const Output& getOutput(void) const
	{
		return output;
	}
	//順伝播
	const DeviceMatrix& forward(const DeviceMatrix& x)
	{
		//inputから順に入れ子に適用
		return output.forward(internal.forward(input.forward(x)));
	}
	//逆伝播
	const DeviceMatrix& back(const DeviceMatrix& d)
	{
		//forwardと逆順に適用
		return input.back(internal.back(output.back(d)));
	}
	//パラメータの更新
	void update(void)
	{
		//TODO 下記の2つの関数を並列化すること
		internal.update(input.getZ());
		output.update(internal.getZ());
	}
	//ミニバッチサイズの取得
	unsigned int getMiniBatchSize(void) const
	{
		return input.getMiniBatchSize();
	}
};

}
