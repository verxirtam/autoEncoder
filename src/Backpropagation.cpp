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
//delta[layer_count - 1] = u[layer_count - 1] - D;
//下記の2式に分けて実行する
//delta[layerCount -1] = u[layer_count - 1];
//delta[layer_count - 1] = (-1.0f) * D + delta[layerCount -1];
void Backpropagation::obtainDeltaLast(const DeviceMatrix& D)
{
	
	//delta[layerCount -1] = u[layer_count - 1];
	delta[layerCount -1] = u[layerCount - 1];
	
	//delta[layer_count - 1] = (-1.0f) * D + delta[layerCount -1];
	float alpha = -1.0f;
	Saxpy(&alpha, D, delta[layerCount -1]);
	
	//ストリーム完了待ち
	CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
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
	Sgemm(&alpha, CUBLAS_OP_T, weight[l + 1], CUBLAS_OP_N, delta[l + 1], &beta, WTdelta[l + 1]);
	
	//ストリーム完了待ち
	CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
	
	//delta[l] = f'(u[l]) ** WTdelta[l + 1];
	obtainDeltaFromFdUWTDelta(l);
	
	//ストリーム完了待ち
	CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
}

void Backpropagation::init(const std::vector<unsigned int>& unit_count, unsigned int minibatch_size)
{
	//NULL Streamを使用する
	CUBLAS_CALL(cublasSetStream(CuBlasManager::getHandle(), 0));
	
	if(layerCount != unit_count.size())
	{
		throw BackpropagationException("error at Backpropagation::init() : layerCount != unit_count.size().");
	}
	unitCount = unit_count;
	
	if(minibatch_size == 0)
	{
		throw BackpropagationException("error at Backpropagation::init() : minibatch_size == 0.");
	}
	miniBatchSize = minibatch_size;
	
	u.clear();
	z.clear();
	weight.clear();
	bias.clear();
	dEdW.clear();
	dEdb.clear();
	WTdelta.clear();
	delta.clear();

	for(unsigned int l = 0; l < layerCount; l++)
	{
		if(unitCount[l] == 0)
		{
			std::stringstream msg;
			msg << "error at Backpropagation::init() : unitCount[" << l << "] == 0.";
			throw BackpropagationException(msg.str());
		}
		
		z.push_back(DeviceMatrix(unitCount[l], miniBatchSize));
		if(l == 0)
		{
			//インデックスl = 0は使用しないのでダミーの値を格納する。
			u.push_back(      DeviceMatrix(1, 1, {0.0f}));
			weight.push_back( DeviceMatrix(1, 1, {0.0f}));
			bias.push_back(   DeviceVector{0.0f}        );
			dEdW.push_back(   DeviceMatrix(1, 1, {0.0f}));
			dEdb.push_back(   DeviceVector{0.0f}        );
			WTdelta.push_back(DeviceMatrix(1, 1, {0.0f}));
			delta.push_back(  DeviceMatrix(1, 1, {0.0f}));
		}
		else
		{
			u.push_back(      DeviceMatrix(unitCount[l], miniBatchSize));
			weight.push_back( DeviceMatrix(unitCount[l],unitCount[l-1]));
			bias.push_back(   DeviceVector(unitCount[l]));
			dEdW.push_back(   DeviceMatrix(unitCount[l],unitCount[l-1]));
			dEdb.push_back(   DeviceVector(unitCount[l]));
			WTdelta.push_back(DeviceMatrix(unitCount[l-1], miniBatchSize));
			delta.push_back(  DeviceMatrix(unitCount[l], miniBatchSize));
		}
	}
	
	_1B = DeviceVector::get1Vector(miniBatchSize);
}

void Backpropagation::initRandom(void)
{
	//NULL Streamを使用する
	CUBLAS_CALL(cublasSetStream(CuBlasManager::getHandle(), 0));
	/*
	std::random_device rdev;
	std::mt19937 engine(rdev());
	std::uniform_real_distribution<float> urd(0.0f, 1.0f);
	*/
	for(DeviceMatrix& w : weight)
	{
		unsigned int M = w.getRowCount();
		unsigned int N = w.getColumnCount();
		/*
		std::vector<float> h_w;
		for(unsigned int i =0; i < M * N; i++)
		{
			//定義域の次元が大きい時に絶対値が大きくなると
			//活性化関数の値が1に上げ止まってしまうので
			//乱数の値をNで割る
			h_w.push_back(urd(engine) / static_cast<float>(N));
		}
		w.set(h_w);
		*/
		
		//wのデバイスメモリに値域(0.0, 1.0]の一様分布に従う乱数を生成
		CURAND_CALL(curandGenerateUniform(CuRandManager::getGenerator(), w.getAddress(), M * N));
		//wを1/Nでスカラー倍する
		float alpha = 1.0f / static_cast<float>(N);
		Sscal(&alpha, w);
		
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
	DeviceMatrix X(x.size(), 1, x);
	DeviceMatrix Y;
	forward(X, Y);
	y = Y.get();
}
void Backpropagation::forward(const DeviceMatrix& X, DeviceMatrix& Y)
{
	//使用するStreamをMainStreamに設定
	//CUBLAS_CALL(cublasSetStream(CuBlasManager::getHandle(), this->getMainStream()));
	CUBLAS_CALL(cublasSetStream(CuBlasManager::getHandle(), 0));
	
	z[0] = X;
	//ストリーム完了待ち
	//CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
	for(unsigned int l = 0; l < layerCount - 1; l++)
	{
		//z[l], weight[l+1], bias[l+1]からu[l+1]を得る
		obtainUFromZ(l);
		//ストリーム完了待ち
		//CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
		//u[l+1]からz[l+1]を得る
		obtainZFromU(l);
		//ストリーム完了待ち
		//CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
	}
	//y = z[L-1]を設定
	Y = z[layerCount - 1];
	//ストリーム完了待ち
	//CUDA_CALL(cudaStreamSynchronize(this->getMainStream()));
}

void Backpropagation::back(const std::vector<float>& d)
{
	DeviceMatrix D(d.size(), 1, d);
	back(D);
}
void Backpropagation::back(const DeviceMatrix& D)
{
	//使用するStreamをMainStreamに設定
	CUBLAS_CALL(cublasSetStream(CuBlasManager::getHandle(), this->getMainStream()));
	
	//初期化
	//delta[layer_count - 1] = u[layer_count - 1] - dd;
	obtainDeltaLast(D);
	
	//lについて降順に逐次実行("**"は要素ごとの積(cudaで実行))
	//delta[l] = f'(u[l]) ** ((W[l + 1])^T * delta[l+1]);
	//l = layerCount - 2, ... , 1
	for(unsigned int l = layerCount - 2; l >= 1; l--)
	{
		obtainDelta(l);
	}
	
	
	//lについて並列実行
	//dEdW[l]  = delta[l] * (z[l - 1])^T;
	//l = layerCount - 1, ... , 1
	for(unsigned int l = layerCount - 1; l >= 1; l--)
	{
		//非同期実行
		obtainDEDW(l);
	}
	unsigned int substream_count = getSubStreamCount();
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
		CUBLAS_CALL(cublasSetStream(CuBlasManager::getHandle(), this->getSubStream(si)));
		
		//W[l] = W[l] - epsilon * dEdW[l]
		//     = W[l] - epsilon * (1 / B) * delta[l] * z[l - 1]^T
		//     = W[l] - (epsilon / B) * delta[l] * z[l - 1]^T
		//     = - (epsilon / B) * delta[l] * z[l - 1]^T + 1.0f * W[l]
		float alpha = - epsilon / miniBatchSize;
		float beta = 1.0f;
		Sgemm(&alpha, CUBLAS_OP_N, delta[l], CUBLAS_OP_T, z[l - 1], &beta, weight[l]);
		
		//W[l] = W[l] - e * dEdW[l];
		//<=> W[l] = - e * dEdW[l] + 1.0f *  W[l];
		//TODO delete after debug
		//alpha = - epsilon;
		//beta = 1.0f;
		//Sgeam(&alpha, CUBLAS_OP_N, dEdW[l], &beta, CUBLAS_OP_N, weight[l], weight[l]);
		
		//使用するStreamを設定
		si = (2 * l + 1) % substream_count;
		CUBLAS_CALL(cublasSetStream(CuBlasManager::getHandle(), this->getSubStream(si)));
		
		//b[l] = b[l] - epsilon * dEdb[l]
		//     = b[l] - epsilon * ((1.0f / B) * delta[l] * _1B)
		//     = b[l] - (epsilon / B) * delta[l] * _1B
		//     = - (epsilon / B) * delta[l] * _1B + 1.0f * b[l]
		alpha = - epsilon / miniBatchSize;
		beta = 1.0f;
		Sgemv(&alpha, CUBLAS_OP_N, delta[l], _1B, &beta, bias[l]);
		
		//b[l] = b[l] - e * dEdb[l];
		//<=> b[l] = - e * dEdb[l] + b[l];
		//TODO delete after debug
		//Saxpy(&alpha, dEdb[l], bias[l]);
	}
	
	//完了待ち
	for(unsigned int s = 0; s < substream_count; s++)
	{
		CUDA_CALL(cudaStreamSynchronize(getSubStream(s)));
	}
}
