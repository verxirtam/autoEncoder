
#pragma once

#include <vector>

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





//(T0, T...)のうちN番目の型 = (T...)のN-1番目の型
template <unsigned int N, class T0, class ... T>
struct layerType
{
	using type = typename layerType<N-1, T...>::type;
};
//0番目の場合はT0
template <class T0, class ... T>
struct layerType<0, T0, T...>
{
	using type = T0;
};

template <class Layer, class ... LayerRemain>
class Serial
{
private:

	Layer layer;
	Serial<LayerRemain...> layerRemain;
	
	
public:
	//N番目のメンバの参照を取得
	template <unsigned int N>
	typename layerType<N, Layer, LayerRemain...>::type& getMember();
};


//N番目のメンバの参照を取得
template <class Layer, class ... LayerRemain>
template <unsigned int N>
typename layerType<N, Layer, LayerRemain...>::type& Serial<Layer, LayerRemain...>::getMember()
{
	return layerRemain.getMember<N-1>();
}

//0番目のメンバ=layerの参照を返す
template <class Layer, class ... LayerRemain>
template <>
typename layerType<0, Layer, LayerRemain...>::type& Serial<Layer, LayerRemain...>::getMember<0>()
{
	return layer;
}



}

