
#pragma once

#include <tuple>

#include "../cuda/DeviceMatrix.h"


namespace nn
{

template <class ActivateMethod, class UpdateMethod>
class TwoLayerPerceptron
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
	TwoLayerPerceptron():
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
};





template <class Layer, class ... LayerRemain>
class Serial
{
private:
	using DeviceMatrix = cuda::DeviceMatrix;
	using DeviceVector = cuda::DeviceVector;
	//レイヤーのタプルの型
	using leyerTupleType = std::tuple<Layer, LayerRemain...>;
	
	using layerCount = std::tuple_size<leyerTupleType>;
	
	//N番目のレイヤーの型
	template <unsigned int N>
	using layerType = typename std::tuple_element<N, leyerTupleType>;
	
	leyerTupleType layer;
	
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
};



}




#include "TwoLayerPerceptron_detail.cuh"


