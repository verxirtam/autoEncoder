#pragma once

#include <vector>
#include <string>

#include "DeviceMatrix.h"

//多クラス分類のための出力層の設定
class OutputLayerMulticlassClassification
{
public:
	//活性化関数 = ソフトマックス関数
	static void activateFunction(const DeviceMatrix& u, DeviceMatrix& z)
	{
		//z_j = exp(u_j) / (Sum_k exp(u_k))
		//TODO 実装すること
		throw 1;
	}
};


