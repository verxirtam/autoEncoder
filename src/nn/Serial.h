
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
	//このクラスの型
	using thisType = Serial<Layer, LayerRemain...>;
	//レイヤのタプルの型
	using layerTupleType = std::tuple<Layer, LayerRemain...>;
	//レイヤの個数
	using layerCount = std::tuple_size<layerTupleType>;
	
	//N番目のレイヤの型
	template <unsigned int N>
	using layerType = typename std::tuple_element<N, layerTupleType>::type;
	//レイヤのタプル
	layerTupleType layer;
	////////////////////////////////////////////////////////
	// forwardMain
	////////////////////////////////////////////////////////
	template <unsigned int N, class T = void>
	struct forwardMain
	{
		static const DeviceMatrix& apply(const DeviceMatrix& x, thisType& tt)
		{
			//N番目のレイヤへの参照
			layerType<N>& l = tt.getMember<N>();
			//N番目のレイヤでforward()実行
			const DeviceMatrix& y = l.forward(x);
			//次のレイヤでforward()実行
			return forwardMain<N + 1>::apply(y, tt);
		}
	};
	//最後((layerCount::value - 1)番目)のレイヤについての特殊化
	template <class T>
	struct forwardMain<layerCount::value - 1, T>
	{
		static const DeviceMatrix& apply(const DeviceMatrix& x, thisType& tt)
		{
			//最後のレイヤへの参照
			layerType<layerCount::value - 1>& l = tt.getMember<layerCount::value - 1>();
			//最後のレイヤでforward()実行
			const DeviceMatrix& y = l.forward(x);
			//実行結果を返却
			return y;
		}
	};
	////////////////////////////////////////////////////////
	// backMain
	////////////////////////////////////////////////////////
	template <unsigned int N, class T = void>
	struct backMain
	{
		static const DeviceMatrix& apply(const DeviceMatrix& weight_t_delta, thisType& tt)
		{
			//N番目のレイヤへの参照
			layerType<N>& l = tt.getMember<N>();
			//N番目のレイヤでback()実行
			const DeviceMatrix& weight_t_delta_out = l.back(weight_t_delta);
			//手前のレイヤでback()実行
			return backMain<N - 1>::apply(weight_t_delta_out, tt);
		}
	};
	//最初(0番目)のレイヤについての特殊化
	template <class T>
	struct backMain<0, T>
	{
		static const DeviceMatrix& apply(const DeviceMatrix& weight_t_delta, thisType& tt)
		{
			//最初のレイヤへの参照
			layerType<0>& l = tt.getMember<0>();
			//最初のレイヤでback()実行
			const DeviceMatrix& weight_t_delta_out = l.back(weight_t_delta);
			//実行結果を返却
			return weight_t_delta_out;
		}
	};
	////////////////////////////////////////////////////////
	// updateMain
	////////////////////////////////////////////////////////
	template <unsigned int N, class T = void>
	struct updateMain
	{
		static void apply(const DeviceMatrix& x, thisType& tt)
		{
			//N番目のレイヤへの参照
			layerType<N>& l = tt.getMember<N>();
			//N番目のレイヤでupdate()実行
			l.update(x);
			//次のレイヤでupdate()実行
			updateMain<N + 1>::apply(l.getZ(), tt);
		}
	};
	//最後((layerCount::value - 1)番目)のレイヤについての特殊化
	template <class T>
	struct updateMain<layerCount::value - 1, T>
	{
		static void apply(const DeviceMatrix& x, thisType& tt)
		{
			//最後のレイヤへの参照
			layerType<layerCount::value - 1>& l = tt.getMember<layerCount::value - 1>();
			//最後のレイヤでupdate()実行
			l.update(x);
		}
	};
	////////////////////////////////////////////////////////
public:
	//N番目のメンバの参照を取得
	template <unsigned int N>
	layerType<N>& getMember()
	{
		return std::get<N>(layer);
	}
	//N番目のメンバの参照を取得(const版)
	template <unsigned int N>
	const layerType<N>& getMember() const
	{
		return std::get<N>(layer);
	}
	const DeviceMatrix& forward(const DeviceMatrix& x)
	{
		return forwardMain<0>::apply(x, *this);
	}
	//逆伝播
	const DeviceMatrix& back(const DeviceMatrix& weight_t_delta)
	{
		return backMain<layerCount::value - 1>::apply(weight_t_delta, *this);
	}
	//パラメータの更新
	void update(const DeviceMatrix& x)
	{
		updateMain<0>::apply(x, *this);
	}
	//zを取得
	const DeviceMatrix& getZ() const
	{
		//最後のレイヤのzを返す
		return getMember<layerCount::value - 1>().getZ();
	}
};



}





