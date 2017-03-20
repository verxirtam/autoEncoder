
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
	__global__
	void invByElement_kernel(float* const w)
	{
		//このスレッドが計算すべき成分のインデックス
		unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
		
		w[j] = 1.0f / w[j];
	}
}



//成分毎に(-1/2)乗を算出する
void Normalization::invSqrtByElement(DeviceVector& W)
{
	{
		invSqrtByElement_kernel<<<W.getDimension(), 1>>>(W.getAddress());
		
		//カーネル実行時のエラーチェック
		CUDA_CALL(cudaGetLastError());
	}
	//TODO 非同期実行されているなら完了待ちしないといけない
	//直後にWを使用するので同期する
	CUDA_CALL(cudaStreamSynchronize(0));
}

//成分毎に(-1)乗を算出する
void Normalization::invByElement(DeviceVector& W)
{
	{
		invByElement_kernel<<<W.getDimension(), 1>>>(W.getAddress());
		
		//カーネル実行時のエラーチェック
		CUDA_CALL(cudaGetLastError());
	}
	//TODO 非同期実行されているなら完了待ちしないといけない
	//直後にWを使用するので同期する
	CUDA_CALL(cudaStreamSynchronize(0));
}

