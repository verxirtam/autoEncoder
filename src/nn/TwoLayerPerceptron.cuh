
#pragma once

#include <tuple>

#include "../cuda/DeviceMatrix.h"

namespace nn
{

template <class ActivateFunction>
class TwoLayerPerceptron
{
private:
	using DeviceMatrix = cuda::DeviceMatrix;
	using DeviceVector = cuda::DeviceVector;
	DeviceMatrix weight;
	DeviceVector bias;
	DeviceMatrix u;
	DeviceMatrix z;
	DeviceMatrix derived_weight;
	DeviceMatrix derived_bias;
	float learningRate;
	float momentum;
public:
	void init(unsigned int dim_input, unsigned int dim_hidden, unsigned int dim_output);
	const DeviceMatrix& forward(DeviceMatrix& x);
	const DeviceMatrix& back(DeviceMatrix& delta);
	void update();
};






template <class Layer, class ... LayerRemain>
class Serial
{
private:
	using DeviceMatrix = cuda::DeviceMatrix;
	using DeviceVector = cuda::DeviceVector;
	//レイヤーのタプルの型
	using leyerTupleType = std::tuple<Layer, LayerRemain...>;
	//N番目のレイヤーの型
	template <unsigned int N>
	using layerType = typename std::tuple_element<N, leyerTupleType>;
	
	leyerTupleType layer;
	
	template <unsigned int N>
	const DeviceMatrix& forwardMain(DeviceMatrix& x)
	{
		auto l = getMember<N>();
		
		auto y = l.forward(x);
		
		if((N + 1) < std::tuple_size<leyerTupleType>::value)
		{
			auto z = forwardMain<N + 1>(y);
			y = z;
		}
		
		return y;
	}
public:
	//N番目のメンバの参照を取得
	template <unsigned int N>
	layerType<N>& getMember()
	{
		return std::get<N>(layer);
	}
	const DeviceMatrix& forward(DeviceMatrix& x)
	{
		return forwardMain<0>(x);
	}
};



}

