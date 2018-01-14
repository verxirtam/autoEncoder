
#pragma once

//成分毎に計算する関数(1変数)
//Func1to1 : 1変数関数

#include "cuda/DeviceMatrix.h"
#include "cuda/DeviceVector.h"


template<class Func1to1>
class ElementWiseFunction1to1
{
	//名前空間cudaを使用
	using DeviceMatrix = cuda::DeviceMatrix;
	using DeviceVector = cuda::DeviceVector;
private:
	static void culculateBlockThreadCount
		(
			const DeviceMatrix& x,
			unsigned int& block_count,
			unsigned int& thread_count,
			unsigned int& thread_count_remain
		);
public:
	//関数の適用
	static DeviceMatrix& apply(const DeviceMatrix& x, DeviceMatrix& y);
};



namespace
{
	template<typename Func1to1>
	__global__
	void apply_kernel
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
		
		y[ij] = Func1to1::apply(x[ij]);
	}
}

template<typename Func1to1>
void ElementWiseFunction1to1<Func1to1>::culculateBlockThreadCount
	(
		const cuda::DeviceMatrix& x,
		unsigned int& block_count,
		unsigned int& thread_count,
		unsigned int& thread_count_remain
	)
{
	//名前空間cudaを使用
	using namespace cuda;
	
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

template<typename Func1to1>
cuda::DeviceMatrix& ElementWiseFunction1to1<Func1to1>::apply(const cuda::DeviceMatrix& x, cuda::DeviceMatrix& y)
{
	//名前空間cudaを使用
	using DeviceMatrix = cuda::DeviceMatrix;
	using DeviceVector = cuda::DeviceVector;
	
	//1ブロックあたりのスレッド数の上限
	static unsigned int thread_count;
	//スレッド数thread_countで実行するブロック数
	unsigned int block_count;
	//スレッド数の残り
	unsigned int thread_count_remain;
	
	ElementWiseFunction1to1<Func1to1>::culculateBlockThreadCount(x, block_count, thread_count, thread_count_remain);
	
	if(block_count != 0)
	{
		//カーネル実行
		apply_kernel<Func1to1><<<block_count, thread_count>>>
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
		apply_kernel<Func1to1><<<1, thread_count_remain>>>
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

