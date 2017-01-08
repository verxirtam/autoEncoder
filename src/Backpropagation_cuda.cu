
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
			unsigned int thread_index_offset,
			const float* const u_lp1,
			float* const z_lp1
		)
	{
		//ブロックインデックス、スレッドインデックスの読み替え
		unsigned int s = threadIdx.x + blockIdx.x * blockDim.x + thread_index_offset;
		
		//このスレッドが計算すべき成分のインデックス
		unsigned int j = s;
		
		//活性化関数の適用
		z_lp1[j] = activateFunction(u_lp1[j]);
	}

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
		
		delta_l[j] = activateFunctionDash(u_l[j]) * wtdelta_lp1[j];
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
	
	//1ブロックあたりのスレッド数の上限
	static unsigned int thread_count = CudaManager::getDeviceProp().maxThreadsPerBlock;
	
	//生成するスレッド数全体
	unsigned int thread_count_total = u[l + 1].getDimension();
	
	//スレッド数thread_countで実行するブロック数
	unsigned int block_count         = thread_count_total / thread_count;
	//スレッド数の残り
	unsigned int thread_count_remain = thread_count_total % thread_count;
	
	if(block_count * thread_count != 0)
	{
		//カーネル実行
		obtainZFromU_kernel<<<block_count, thread_count>>>
			(
				0,
				u[l + 1].getAddress(),
				z[l + 1].getAddress()
			);
		//カーネル実行時のエラーチェック
		CUDA_CALL(cudaGetLastError());
	}
	if(thread_count_remain != 0)
	{
		//カーネル実行
		obtainZFromU_kernel<<<1, thread_count_remain>>>
			(
				block_count * thread_count,
				u[l + 1].getAddress(),
				z[l + 1].getAddress()
			);
		//カーネル実行時のエラーチェック
		CUDA_CALL(cudaGetLastError());
	}
	
	//直後にu[l + 1]を使用するので同期する
	CUDA_CALL(cudaStreamSynchronize(0));
}


//delta[l] = f'(u[l]) ** WTdelta[l + 1];
void Backpropagation::obtainDeltaFromFdUWTDelta(unsigned int l)
{
	//1ブロックあたりのスレッド数の上限
	static unsigned int thread_count = CudaManager::getDeviceProp().maxThreadsPerBlock;
	
	//生成するスレッド数全体
	unsigned int thread_count_total = u[l].getDimension();
	
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

void Backpropagation::obtainDEDW(unsigned int l, unsigned int thread_count, unsigned int d)
{
	switch(d)
	{
	case 1:
		obtainDEDWMain<1>(l, thread_count);
		break;
	case 32:
		obtainDEDWMain<32>(l, thread_count);
		break;
	case 64:
		obtainDEDWMain<64>(l, thread_count);
		break;
	case 128:
		obtainDEDWMain<128>(l, thread_count);
		break;
	case 256:
		obtainDEDWMain<256>(l, thread_count);
		break;
	case 512:
		obtainDEDWMain<512>(l, thread_count);
		break;
	case 1024:
		obtainDEDWMain<1024>(l, thread_count);
		break;
	}
}
//dEdW[l] = delta[l] * (z[l - 1])^T;
//D:1スレッドが処理する変数の個数
template<unsigned int D>
void Backpropagation::obtainDEDWMain(unsigned int l, unsigned int thread_count)
{
	//実行するStream
	unsigned int substream_count = this->getSubStreamCount();
	cudaStream_t si0 = this->getSubStream((l + 0) % substream_count);
	cudaStream_t si1 = this->getSubStream((l + 1) % substream_count);
	cudaStream_t si2 = this->getSubStream((l + 2) % substream_count);
	
	unsigned int N = dEdW[l].getRowCount();
	unsigned int M = dEdW[l].getColumnCount();
	//実行するスレッド数の合計
	unsigned int data_count_total = N * M;
	
	//ブロック数
	unsigned int block_count0 = data_count_total / (D * thread_count);
	//スレッド数の残り
	unsigned int remain0      = data_count_total % (D * thread_count);
	//ブロック数
	unsigned int block_count1 = remain0 / thread_count;
	//スレッド数の残り
	unsigned int remain1      = remain0 % thread_count;
	
	if(block_count0 != 0)
	{
		//ブロックあたりthread_countスレッドで実行
		//dEdW[l] = delta[l] * (z[l - 1])^T;
		obtainDEDW_kernel<D><<<block_count0, thread_count, 0, si0>>>
			(
				0,
				N,
				delta[l].getAddress(),
				z[l - 1].getAddress(),
				dEdW[l].getAddress()
			);
		//カーネル実行時のエラーチェック
		CUDA_CALL(cudaGetLastError());
	}
	if(remain0 != 0)
	{
		//上記のカーネル実行で余ったスレッドの実行
		if(block_count1 != 0)
		{
			//dEdW[l] = delta[l] * (z[l - 1])^T;
			obtainDEDW_kernel<1><<<block_count1, thread_count, 0, si1>>>
				(
					D * block_count0 * thread_count,
					N,
					delta[l].getAddress(),
					z[l - 1].getAddress(),
					dEdW[l].getAddress()
				);
			//カーネル実行時のエラーチェック
			CUDA_CALL(cudaGetLastError());
		}
		if(remain1 != 0)
		{
			//dEdW[l] = delta[l] * (z[l - 1])^T;
			obtainDEDW_kernel<1><<<1, remain1, 0, si2>>>
				(
					D * block_count0 * thread_count + block_count1 * thread_count,
					N,
					delta[l].getAddress(),
					z[l - 1].getAddress(),
					dEdW[l].getAddress()
				);
			//カーネル実行時のエラーチェック
			CUDA_CALL(cudaGetLastError());
		}
	}
}


