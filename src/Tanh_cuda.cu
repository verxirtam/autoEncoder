
#include "Tanh.h"

namespace
{
	__host__ __device__
	float activate(float x)
	{
		return tanhf(x);
	}
	__global__
	void activate_kernel
		(
			unsigned int thread_index_offset,
			const float* const x,
			float* const y
		)
	{
		//ブロックインデックス、スレッドインデックスの読み替え
		unsigned int s = threadIdx.x + blockIdx.x * blockDim.x + thread_index_offset;
		
		//このスレッドが計算すべき成分のインデックス
		unsigned int ij = s;
		
		y[ij] = activate(x[ij]);
	}
}

DeviceMatrix& Tanh::activate(const DeviceMatrix& x, DeviceMatrix& y)
{
	//1ブロックあたりのスレッド数の上限
	static unsigned int thread_count = CudaManager::getDeviceProp().maxThreadsPerBlock;
	
	//生成するスレッド数全体
	unsigned int thread_count_total = x.getRowCount() * x.getColumnCount();
	
	//スレッド数thread_countで実行するブロック数
	unsigned int block_count         = thread_count_total / thread_count;
	//スレッド数の残り
	unsigned int thread_count_remain = thread_count_total % thread_count;
	
	if(block_count != 0)
	{
		//カーネル実行
		activate_kernel<<<block_count, thread_count>>>
			(
				0,
				x.getAddress(),
				y.getAddress()
			);
		//カーネル実行時のエラーチェック
		CUDA_CALL(cudaGetLastError());
	}
	if(thread_count_remain != 0)
	{
		//カーネル実行
		activate_kernel<<<1, thread_count_remain>>>
			(
				block_count * thread_count,
				x.getAddress(),
				y.getAddress()
			);
		//カーネル実行時のエラーチェック
		CUDA_CALL(cudaGetLastError());
	}
	//直後にdelta[l]を使用するので同期する
	CUDA_CALL(cudaStreamSynchronize(0));
	
	return y;
}

