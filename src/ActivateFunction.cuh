#pragma once

#include "cuda/DeviceMatrix.h"

#include "ElementWiseFunction1to1.cuh"
#include "ElementWiseFunctionUtil.cuh"

template<typename Func1to1>
class ActivateFunction
{
public:
	//活性化関数
	static DeviceMatrix& activate(const DeviceMatrix& x, DeviceMatrix& y)
	{
		return ElementWiseFunction1to1<Func1to1>::apply(x, y);
	}
	//活性化関数の微分
	static DeviceMatrix& activateDiff(const DeviceMatrix& x, DeviceMatrix& y)
	{
		return ElementWiseFunction1to1<Func1to1ApplyDiff<Func1to1> >::apply(x, y);
	}
};

