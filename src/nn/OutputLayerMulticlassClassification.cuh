#pragma once

#include <vector>
#include <string>

#include "DeviceMatrix.h"

#include "ElementWiseFunction1to1.cuh"
#include "Func1to1Exp.cuh"

namespace nn
{


//多クラス分類のための出力層の設定
class OutputLayerMulticlassClassification
{
public:
	//活性化関数 = ソフトマックス関数
	static void activateFunction(const DeviceMatrix& u, DeviceMatrix& z)
	{
		//z_j = exp(u_j) / (Sum_k exp(u_k))
		//z = (exp(u_j))_j
		ElementWiseFunction1to1<Func1to1Exp>::apply(u, z);
		
		//TODO 実装すること(成分の和で割る関数)
		
		throw 1;
	}
};

}

