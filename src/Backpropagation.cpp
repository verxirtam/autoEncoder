/*
 * =====================================================================================
 *
 *       Filename:  Backpropagation.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月19日 03時55分38秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "Backpropagation.h"


//初期化
//下記を実行する
//delta[layer_count - 1] = u[layer_count - 1] - dd;
//下記の2式に分けて実行する
//delta[layerCount -1] = u[layer_count - 1];
//delta[layer_count - 1] = (-1.0f) * dd + delta[layerCount -1];
void Backpropagation::obtainDeltaLast(const std::vector<float>& d)
{
	DeviceVector dd(d);
	
	//delta[layerCount -1] = u[layer_count - 1];
	delta[layerCount -1] = u[layerCount - 1];
	
	//delta[layer_count - 1] = (-1.0f) * dd + delta[layerCount -1];
	float alpha = -1.0f;
	Saxpy(&alpha, dd, delta[layerCount -1]);
}

//逆伝播でのdeltaの算出
//lについて降順に逐次実行("**"は要素ごとの積(cudaで実行))
//delta[l] = f'(u[l]) ** ((W[l + 1])^T * delta[l+1]);
//l = layerCount - 2, ... , 1
void Backpropagation::obtainDelta(unsigned int l)
{
	//WTdelta[l +1] = (W[l + 1])^T * delta[l+1];
	//<=>
	//WTdelta[l +1] = 1.0f * (W[l + 1])^T * delta[l+1] + 0.0f * WTdelta[l +1];
	float alpha = 1.0f;
	float beta  = 0.0f;
	Sgemv(&alpha, CUBLAS_OP_T, weight[l + 1], delta[l + 1], &beta, WTdelta[l + 1]);
	
	//ストリーム完了待ち
	CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
	
	//delta[l] = f'(u[l]) ** WTdelta[l + 1];
	obtainDeltaFromFdUWTDelta(l);
	
	//ストリーム完了待ち
	CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
}

void Backpropagation::init(const std::vector<unsigned int>& unit_count)
{
	if(layerCount != unit_count.size())
	{
		throw BackpropagationException("error at Backpropagation::init() : layerCount != unit_count.size().");
	}
	unitCount = unit_count;

	u.clear();
	z.clear();
	weight.clear();
	bias.clear();
	dEdW.clear();
	dEdb.clear();

	for(unsigned int l = 0; l < layerCount; l++)
	{
		if(unitCount[l] == 0)
		{
			std::stringstream msg;
			msg << "error at Backpropagation::init() : unitCount[" << l << "] == 0.";
			throw BackpropagationException(msg.str());
		}
		
		z.push_back(DeviceVector(unitCount[l]));
		if(l == 0)
		{
			//インデックスl = 0は使用しないのでダミーの値を格納する。
			u.push_back(      DeviceVector{0.0f}      );
			weight.push_back( DeviceMatrix(1,1,{0.0f}));
			bias.push_back(   DeviceVector{0.0f}      );
			dEdW.push_back(   DeviceMatrix(1,1,{0.0f}));
			dEdb.push_back(   DeviceVector{0.0f}      );
			WTdelta.push_back(DeviceVector{0.0f}      );
		}
		else
		{
			u.push_back(     DeviceVector(unitCount[l]));
			weight.push_back(DeviceMatrix(unitCount[l],unitCount[l-1]));
			bias.push_back(  DeviceVector(unitCount[l]));
			dEdW.push_back(  DeviceMatrix(unitCount[l],unitCount[l-1]));
			dEdb.push_back(  DeviceVector(unitCount[l]));
			WTdelta.push_back(DeviceVector(unitCount[l-1]));
		}
	}
}

void Backpropagation::initRandom(void)
{
	std::random_device rdev;
	std::mt19937 engine(rdev());
	std::uniform_real_distribution<float> urd(0.0f, 1.0f);

	for(auto&& w : weight)
	{
		unsigned int M = w.getRowCount();
		unsigned int N = w.getColumnCount();
		std::vector<float> h_w;
		for(unsigned int i =0; i < M * N; i++)
		{
			//定義域の次元が大きい時に絶対値が大きくなると
			//活性化関数の値が1に上げ止まってしまうので
			//乱数の値をNで割る
			h_w.push_back(urd(engine) / static_cast<float>(N));
		}
		w.set(h_w);
	}
	//biasは一律0で初期化する
	for(auto&& b : bias)
	{
		unsigned int N = b.getDimension();
		std::vector<float> h_b;
		for(unsigned int i =0; i < N; i++)
		{
			h_b.push_back(0.0f);
		}
		b.set(h_b);
	}
}

void Backpropagation::forward(const std::vector<float>& x, std::vector<float>& y)
{
	//使用するStreamをMainStreamに設定
	CUBLAS_CALL(cublasSetStream(CUBLASManager::getHandle(), this->getMainStream()));
	
	z[0].set(x);
	//ストリーム完了待ち
	CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
	for(unsigned int l = 0; l < layerCount - 1; l++)
	{
		//z[l], weight[l+1], bias[l+1]からu[l+1]を得る
		obtainUFromZ(l);
		//ストリーム完了待ち
		CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
		//u[l+1]からz[l+1]を得る
		obtainZFromU(l);
		//ストリーム完了待ち
		CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
	}
	//y = z[L-1]を設定
	y = z[layerCount - 1].get();
	//ストリーム完了待ち
	CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
}

void Backpropagation::back(const std::vector<float>& d)
{
	//使用するStreamをMainStreamに設定
	CUBLAS_CALL(cublasSetStream(CUBLASManager::getHandle(), this->getMainStream()));
	
	//初期化
	//delta[layer_count - 1] = u[layer_count - 1] - dd;
	obtainDeltaLast(d);
	
	//lについて降順に逐次実行("**"は要素ごとの積(cudaで実行))
	//delta[l] = f'(u[l]) ** ((W[l + 1])^T * delta[l+1]);
	//l = layerCount - 2, ... , 1
	for(unsigned int l = layerCount - 2; l >= 1; l--)
	{
		obtainDelta(l);
	}
	
	
	unsigned int substream_count = getSubStreamCount();
	//lについて並列実行
	//dEdW[l]  = delta[l] * (z[l - 1])^T;
	//l = layerCount - 1, ... , 1
	for(unsigned int l = layerCount - 1; l >= 1; l--)
	{
		//非同期実行
		obtainDEDW(l, l % substream_count);
	}
	//完了待ち
	for(unsigned int s = 0; s < substream_count; s++)
	{
		CUDA_CALL(cudaStreamSynchronize(getSubStream(s)));
	}
	
	//対応不要
	//dEdb[l]  = delta[l];
	//l = layerCount - 1, ... , 2
	
}

void Backpropagation::updateParameter()
{
	//l,W,b について非同期に実行
	
	unsigned int substream_count = getSubStreamCount();
	
	for(unsigned int l = 1; l < layerCount; l++)
	{
		//使用するStreamを設定
		unsigned int si = (2 * l) % substream_count;
		CUBLAS_CALL(cublasSetStream(CUBLASManager::getHandle(), this->getSubStream(si)));
		
		//W[l] = W[l] - e * dEdW[l];
		//<=> W[l] = - e * dEdW[l] + 1.0f *  W[l];
		float alpha = - epsilon;
		float beta = 1.0f;
		Sgeam(&alpha, CUBLAS_OP_N, dEdW[l], &beta, CUBLAS_OP_N, weight[l], weight[l]);
		
		//使用するStreamを設定
		si = (2 * l + 1) % substream_count;
		CUBLAS_CALL(cublasSetStream(CUBLASManager::getHandle(), this->getSubStream(si)));
		
		//b[l] = b[l] - e * dEdb[l];
		//<=> b[l] = - e * dEdb[l] + b[l];
		Saxpy(&alpha, dEdb[l], bias[l]);
	}
	
	//完了待ち
	for(unsigned int s = 0; s < substream_count; s++)
	{
		CUDA_CALL(cudaStreamSynchronize(getSubStream(s)));
	}
}
