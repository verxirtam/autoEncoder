
#pragma once

#include <tuple>

#include "../cuda/DeviceMatrix.h"


namespace nn
{

template <class Layer, class ... LayerRemain>
class Serial
{
private:
	using DeviceMatrix = cuda::DeviceMatrix;
	using DeviceVector = cuda::DeviceVector;
	//レイヤーのタプルの型
	using layerTupleType = std::tuple<Layer, LayerRemain...>;
	
	using layerCount = std::tuple_size<layerTupleType>;
	
	//N番目のレイヤーの型
	template <unsigned int N>
	using layerType = typename std::tuple_element<N, layerTupleType>;
	
	layerTupleType layer;
	
	template <unsigned int N>
	const DeviceMatrix& forwardMain(DeviceMatrix& x)
	{
		auto l = getMember<N>();
		
		auto y = l.forward(x);
		
		if((N + 1) < layerCount::value)
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
	//逆伝播
	const DeviceMatrix& back(const DeviceMatrix& delta_output);
	//パラメータの更新
	void update(const DeviceMatrix& x);
	//zを取得
	const DeviceMatrix& getZ() const
	{
		//最後のレイヤのzを返す
		return getMember<layerCount::value - 1>().getZ();
	}
};



}


