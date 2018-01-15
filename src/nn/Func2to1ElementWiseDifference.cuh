
#pragma once

namespace nn
{


class Func2to1ElementWiseDifference
{
public:
	//関数
	__host__ __device__
	static float apply(float x, float y)
	{
		return x - y;
	}
};

}

