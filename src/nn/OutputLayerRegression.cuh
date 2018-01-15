#pragma once

#include <vector>
#include <string>

#include "../cuda/DeviceMatrix.h"

namespace nn
{

//回帰問題のための出力層の設定
class OutputLayerRegression
{
public:
	//活性化関数 = 恒等写像
	static void activateFunction(const cuda::DeviceMatrix& u, cuda::DeviceMatrix& z)
	{
		z = u;
	}
};

}

