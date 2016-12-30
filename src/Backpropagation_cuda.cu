
#include "Backpropagation.h"

namespace
{
	__host__ __device__
	float activateFunction(float x)
	{
		return tanhf(x);
	}
	
	__host__ __device__
	float activateFunctionDash(float x)
	{
		float tanh_x = tanhf(x);
		return 1.0f - (tanh_x * tanh_x);
	}
	
	
	__global__
	void obtainZFromU_kernel
		(
			const float* const u_lp1,
			float* const z_lp1
		)
	{
		//ブロックインデックス、スレッドインデックスの読み替え
		int tx = threadIdx.x;
		int txsize = blockDim.x;
		int bx = blockIdx.x;
		
		//このスレッドが計算すべき成分のインデックス
		int j = tx + bx * txsize;
		
		//活性化関数の適用
		z_lp1[j] = activateFunction(u_lp1[j]);
	}

	__global__
	void obtainDeltaFromFdUWTDelta_kernel
		(
			const float* const u_l,
			const float* const wtdelta_lp1,
			float* const delta_l
		)
	{
		//ブロックインデックス、スレッドインデックスの読み替え
		int tx = threadIdx.x;
		int txsize = blockDim.x;
		int bx = blockIdx.x;
		
		//このスレッドが計算すべき成分のインデックス
		int j = tx + bx * txsize;
		
		delta_l[j] = activateFunctionDash(u_l[j]) * wtdelta_lp1[j];
	}
	
	//dEdW[l] = delta[l] * (z[l])^T;
	__global__
	void obtainDEDW_kernel
		(
			unsigned int block_index_offset,
			unsigned int thread_count,
			unsigned int row_count,
			const float* const delta_l,
			const float* const z_lm1,
			float* const dedw_l
		)
	{
		//ブロックインデックス、スレッドインデックスの読み替え
		unsigned int s = threadIdx.x + blockIdx.x * blockDim.x + block_index_offset * thread_count;
		unsigned int i = s % row_count;
		unsigned int j = s / row_count;
		unsigned int k = s;
		
		dedw_l[k] = delta_l[i] * z_lm1[j];
	}
}

void Backpropagation::obtainZFromU(unsigned int l)
{
	//最後のレイヤの場合は恒等写像なので単にコピーする
	if(l == layerCount - 1)
	{
		z[l + 1] = u[l + 1];
		return;
	}
	
	int block_count = 1;
	int thread_count = u[l + 1].getDimension();
	obtainZFromU_kernel<<<block_count, thread_count>>>(u[l + 1].getAddress(), z[l + 1].getAddress());
	//カーネル実行時のエラーチェック
	CUDA_CALL(cudaGetLastError());
	//直後にu[l + 1]を使用するので同期する
	//cudaDeviceSynchronize();
	CUDA_CALL(cudaStreamSynchronize(0));
}


//delta[l] = f'(u[l]) ** WTdelta[l + 1];
void Backpropagation::obtainDeltaFromFdUWTDelta(unsigned int l)
{
	int block_count = 1;
	int thread_count = u[l].getDimension();
	obtainDeltaFromFdUWTDelta_kernel<<<block_count, thread_count>>>
		(
			u[l].getAddress(),
			WTdelta[l + 1].getAddress(),
			delta[l].getAddress()
		);
	//カーネル実行時のエラーチェック
	CUDA_CALL(cudaGetLastError());
	//直後にdelta[l]を使用するので同期する
	//cudaDeviceSynchronize();
	CUDA_CALL(cudaStreamSynchronize(0));
}

//dEdW[l] = delta[l] * (z[l - 1])^T;
void Backpropagation::obtainDEDW(unsigned int l)
{
	//1ブロックあたりのスレッド数の上限(GTX1080の値)
	unsigned int thread_count = 1024;
	
	unsigned int N = dEdW[l].getRowCount();
	unsigned int M = dEdW[l].getColumnCount();
	//実行するスレッド数の合計
	unsigned int thread_count_total = N * M;
	
	//スレッド数thread_countで実行するブロック数
	unsigned int block_count         = thread_count_total / thread_count;
	//スレッド数の残り
	unsigned int thread_count_remain = thread_count_total % thread_count;
	
	if(block_count * thread_count != 0)
	{
		//ブロックあたりthread_countスレッドで実行
		//dEdW[l] = delta[l] * (z[l - 1])^T;
		obtainDEDW_kernel<<<block_count, thread_count>>>
			(
				0,
				thread_count,
				N,
				delta[l].getAddress(),
				z[l - 1].getAddress(),
				dEdW[l].getAddress()
			);
		//カーネル実行時のエラーチェック
		CUDA_CALL(cudaGetLastError());
	}
	if(thread_count_remain != 0)
	{
		//上記のカーネル実行で余ったスレッドの実行
		//dEdW[l] = delta[l] * (z[l - 1])^T;
		obtainDEDW_kernel<<<1, thread_count_remain>>>
			(
				block_count,
				thread_count,
				N,
				delta[l].getAddress(),
				z[l - 1].getAddress(),
				dEdW[l].getAddress()
			);
		//カーネル実行時のエラーチェック
		CUDA_CALL(cudaGetLastError());
	}
}


