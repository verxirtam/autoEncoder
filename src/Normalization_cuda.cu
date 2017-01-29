
#include "Normalization.h"

namespace
{
	__global__
	void invSqrtByElement_kernel(float* const w)
	{
		//このスレッドが計算すべき成分のインデックス
		unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
		
		w[j] = 1.0f / sqrtf(w[j]);
	}
}



//成分毎に(-1/2)乗を算出する
void Normalization::invSqrtByElement(DeviceVector& W)
{
	invSqrtByElement_kernel<<<W.getDimension(), 1>>>(W.getAddress());
}


