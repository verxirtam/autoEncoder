
#include "Backpropagation.h"

#include "ActivateFunction.cuh"
#include "Tanh.cuh"

namespace
{
	__global__
	void obtainDeltaFromFdUWTDelta_kernel
		(
			unsigned int thread_index_offset,
			const float* const u_l,
			const float* const wtdelta_lp1,
			float* const delta_l
		)
	{
		//ブロックインデックス、スレッドインデックスの読み替え
		unsigned int s = threadIdx.x + blockIdx.x * blockDim.x + thread_index_offset;
		
		//このスレッドが計算すべき成分のインデックス
		unsigned int j = s;
		
		delta_l[j] = Tanh::applyDiff(u_l[j]) * wtdelta_lp1[j];
	}
	
	//dEdW[l] = delta[l] * (z[l])^T;
	template<unsigned int D>
	__global__
	void obtainDEDW_kernel
		(
			unsigned int data_index_offset,
			unsigned int row_count,
			const float* const delta_l,
			const float* const z_lm1,
			float* const dedw_l
		)
	{
		for(unsigned int d =0; d < D; d++)
		{
			//ブロックインデックス、スレッドインデックスの読み替え
			unsigned int s = threadIdx.x + d * blockDim.x + blockIdx.x * D * blockDim.x + data_index_offset;
			unsigned int i = s % row_count;
			unsigned int j = s / row_count;
			unsigned int k = s;
			
			//dEdW[l] = delta[l] * (z[l])^T;
			dedw_l[k] = delta_l[i] * z_lm1[j];
		}
	}
}

void Backpropagation::obtainZFromU(unsigned int l)
{
	
	//最後のレイヤ(l == layerCount - 2)の場合は恒等写像なので単にコピーする
	if(l == layerCount - 2)
	{
		z[l + 1] = u[l + 1];
		return;
	}
	
	ActivateFunction<Tanh>::activate(u[l + 1], z[l + 1]);
}


//delta[l] = f'(u[l]) ** WTdelta[l + 1];
void Backpropagation::obtainDeltaFromFdUWTDelta(unsigned int l)
{
	//1ブロックあたりのスレッド数の上限
	static unsigned int thread_count = CudaManager::getDeviceProp().maxThreadsPerBlock;
	
	//生成するスレッド数全体
	unsigned int thread_count_total = u[l].getRowCount() * u[l].getColumnCount();
	
	//スレッド数thread_countで実行するブロック数
	unsigned int block_count         = thread_count_total / thread_count;
	//スレッド数の残り
	unsigned int thread_count_remain = thread_count_total % thread_count;
	
	if(block_count * thread_count != 0)
	{
		//カーネル実行
		obtainDeltaFromFdUWTDelta_kernel<<<block_count, thread_count>>>
			(
				0,
				u[l].getAddress(),
				WTdelta[l + 1].getAddress(),
				delta[l].getAddress()
			);
		//カーネル実行時のエラーチェック
		CUDA_CALL(cudaGetLastError());
	}
	if(thread_count_remain != 0)
	{
		//カーネル実行
		obtainDeltaFromFdUWTDelta_kernel<<<1, thread_count_remain>>>
			(
				block_count * thread_count,
				u[l].getAddress(),
				WTdelta[l + 1].getAddress(),
				delta[l].getAddress()
			);
		//カーネル実行時のエラーチェック
		CUDA_CALL(cudaGetLastError());
	}
	//直後にdelta[l]を使用するので同期する
	CUDA_CALL(cudaStreamSynchronize(0));
}



