
#include "Normalization.h"

namespace nn
{

namespace
{
	__global__
	void invSqrtByElement_kernel(float* const w)
	{
		//このスレッドが計算すべき成分のインデックス
		unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
		
		float sqrt_w_j = sqrtf(w[j]);
		
		//逆数を計算する絶対値の上限(1/(2^23))
		const float epsilon_inv = 8388608.0f;
		const float epsilon     = 1.0f / epsilon_inv;
		
		//w[j] = 1.0f / sqrtf(w[j]);
		w[j] = (sqrt_w_j >= epsilon) ? (1.0f / sqrt_w_j) : (epsilon_inv);
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

}

