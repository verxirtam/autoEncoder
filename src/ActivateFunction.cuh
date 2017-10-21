#pragma once

#include "DeviceMatrix.h"


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
	static DeviceMatrix& activate(const DeviceMatrix& x, DeviceMatrix& y);
	//活性化関数の微分
	static DeviceMatrix& activateDiff(const DeviceMatrix& x, DeviceMatrix& y);
};

//TODO activate()向けとactivateDiff()向けを2重に書いているのを統合する


namespace
{
	template<typename Func>
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
		
		y[ij] = Func::apply(x[ij]);
	}
	template<typename Func>
	__global__
	void activateDiff_kernel
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
		
		y[ij] = Func::applyDiff(x[ij]);
	}
}

template<typename Func>
void ActivateFunction<Func>::culculateBlockThreadCount
	(
		const DeviceMatrix& x,
		unsigned int& block_count,
		unsigned int& thread_count,
		unsigned int& thread_count_remain
	)
{
	//1ブロックあたりのスレッド数の上限
	static unsigned int thread_count_local = CudaManager::getDeviceProp().maxThreadsPerBlock;
	thread_count = thread_count_local;
	
	//生成するスレッド数全体
	unsigned int thread_count_total = x.getRowCount() * x.getColumnCount();
	
	//スレッド数thread_countで実行するブロック数
	block_count         = thread_count_total / thread_count;
	//スレッド数の残り
	thread_count_remain = thread_count_total % thread_count;
}

template<typename Func>
DeviceMatrix& ActivateFunction<Func>::activate(const DeviceMatrix& x, DeviceMatrix& y)
{
	//1ブロックあたりのスレッド数の上限
	static unsigned int thread_count;
	//スレッド数thread_countで実行するブロック数
	unsigned int block_count;
	//スレッド数の残り
	unsigned int thread_count_remain;
	
	ActivateFunction::culculateBlockThreadCount(x, block_count, thread_count, thread_count_remain);
	
	if(block_count != 0)
	{
		//カーネル実行
		activate_kernel<Func><<<block_count, thread_count>>>
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
		activate_kernel<Func><<<1, thread_count_remain>>>
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



template<typename Func>
DeviceMatrix& ActivateFunction<Func>::activateDiff(const DeviceMatrix& x, DeviceMatrix& y)
{
	//1ブロックあたりのスレッド数の上限
	static unsigned int thread_count;
	//スレッド数thread_countで実行するブロック数
	unsigned int block_count;
	//スレッド数の残り
	unsigned int thread_count_remain;
	
	ActivateFunction::culculateBlockThreadCount(x, block_count, thread_count, thread_count_remain);
	
	if(block_count != 0)
	{
		//カーネル実行
		activateDiff_kernel<Func><<<block_count, thread_count>>>
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
		activateDiff_kernel<Func><<<1, thread_count_remain>>>
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

