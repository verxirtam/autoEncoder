
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
	void obtainZFromU_Kernel
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
			const float* const delta_l,
			const float* const z_lm1,
			float* const dedw_l
		)
	{
		//ブロックインデックス、スレッドインデックスの読み替え
		int i = threadIdx.x;
		int imax = blockDim.x;
		int j = threadIdx.y;
		
		dedw_l[i + j * imax] = delta_l[i] * z_lm1[j];
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
	obtainZFromU_Kernel<<<block_count, thread_count>>>(u[l + 1].getAddress(), z[l + 1].getAddress());
	//直後にu[l + 1]を使用するので同期する
	cudaDeviceSynchronize();
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
	//直後にdelta[l]を使用するので同期する
	cudaDeviceSynchronize();
}

//dEdW[l] = delta[l] * (z[l - 1])^T;
void Backpropagation::obtainDEDW(unsigned int l)
{
	dim3 grid(1, 1, 1);
	dim3 block(1, 1, 1);
	block.x = dEdW[l].getRowCount();
	block.y = dEdW[l].getColumnCount();
	
	//dEdW[l] = delta[l] * (z[l - 1])^T;
	obtainDEDW_kernel<<<grid, block>>>(delta[l].getAddress(), z[l - 1].getAddress(), dEdW[l].getAddress());
}


