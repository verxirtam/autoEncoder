#pragma once

#include "DeviceMatrix.h"

#include "ElementWiseFunction1_1.cuh"
#include "ElementWiseFunctionUtil.cuh"

template<typename Func>
class ActivateFunction
{
private:
	static void culculateBlockThreadCount
		(
			const DeviceMatrix& x,
			unsigned int& block_count,
			unsigned int& thread_count,
			unsigned int& thread_count_remain
		);
public:
	//活性化関数
	static DeviceMatrix& activate(const DeviceMatrix& x, DeviceMatrix& y)
	{
		return ElementWiseFunction1_1<Func>::apply(x, y);
	}
	//活性化関数の微分
	static DeviceMatrix& activateDiff(const DeviceMatrix& x, DeviceMatrix& y)
	{
		return ElementWiseFunction1_1<ApplyDiff1_1<Func> >::apply(x, y);
	}
};

