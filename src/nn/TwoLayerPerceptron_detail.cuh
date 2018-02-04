/*
 * =====================================================================================
 *
 *       Filename:  TwoLayerPerceptron_detail.cuh
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年01月30日 23時05分02秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

namespace nn
{


//初期化
template <class ActivateMethod, class UpdateMethod>
void TwoLayerPerceptron<ActivateMethod, UpdateMethod>::init
	(
		unsigned int dim_input,
		unsigned int dim_output,
		unsigned int minibatch_size
	)
{
	//パラメータの初期化
	miniBatchSize = minibatch_size;
	weight = DeviceMatrix(dim_output, dim_input);
	bias   = DeviceVector(dim_output);
	u      = DeviceMatrix(dim_output, miniBatchSize);
	z      = DeviceMatrix(dim_output, miniBatchSize);
	_1B    = DeviceVector::get1Vector(miniBatchSize);
	delta  = DeviceMatrix(dim_output, miniBatchSize);
	
	//weightとbiasをランダムに初期化
	initWeightBias();
	

	//パラメータ更新手法の初期化
	updateMethod.init(dim_input, dim_output, minibatch_size);
}

//weight, biasをランダムに初期化する
template <class ActivateMethod, class UpdateMethod>
void TwoLayerPerceptron<ActivateMethod, UpdateMethod>::initWeightBias(void)
{
	//weightの初期化
	{
		unsigned int N = weight.getRowCount();
		unsigned int M = weight.getColumnCount();
		//平均
		float mean = 0.0f;
		//標準偏差
		float stddev = 0.125f / std::sqrt(static_cast<float>(N));
		//指定した平均、標準偏差の正規分布に従った確率変数でweightの各成分を初期化する
		CURAND_CALL(curandGenerateNormal(cuda::CuRandManager::getGenerator(),weight.getAddress(), M * N, mean, stddev));
	}
	//biasの初期化
	{
		unsigned int N = bias.getDimension();
		bias.set(std::vector<float>(N, 0.0f));
	}
}


//順伝播
template <class ActivateMethod, class UpdateMethod>
const cuda::DeviceMatrix& TwoLayerPerceptron<ActivateMethod, UpdateMethod>::forward(const DeviceMatrix& x)
{
	//weight, biasを掛ける
	//u = weight * x + bias * _1B ^ T
	forwardLinear(x);
	
	//活性化関数を適用
	//z = ActivateMethod(u) : ElementWiseFunctionとは限らない
	ActivateMethod::activate(u, z);
	
	return z;
}

//順伝播の線型部分
//u = weight * x + bias * _1B ^ T
template <class ActivateMethod, class UpdateMethod>
void TwoLayerPerceptron<ActivateMethod, UpdateMethod>::forwardLinear(const DeviceMatrix& x)
{
	
	//u = weight * x;
	//  = 1.0f * weight * x + 0.0f * u;
	float alpha = 1.0f;
	float beta  = 0.0f;
	Sgemm(&alpha, CUBLAS_OP_N, weight, CUBLAS_OP_N, x, &beta, u);
	
	//u = 1.0f * bias * _1B ^ T + u;
	//<=>
	//u = weight * x + bias * _1B ^ T;
	alpha = 1.0;
	Sger(&alpha, bias, _1B, u);
}

//weightTDeltaを算出する
//weightTDelta = weight^T * delta;
template <class ActivateMethod, class UpdateMethod>
void TwoLayerPerceptron<ActivateMethod, UpdateMethod>::getWeightTDelta()
{
	//weightTdelta = (weight)^T * delta;
	//<=>
	//weightTdelta = 1.0f * (weight)^T * delta + 0.0f * weightTdelta;
	float alpha = 1.0f;
	float beta  = 0.0f;
	Sgemm(&alpha, CUBLAS_OP_T, weight, CUBLAS_OP_N, delta, &beta, weightTDelta);
	
}

//逆伝播
template <class ActivateMethod, class UpdateMethod>
const cuda::DeviceMatrix& TwoLayerPerceptron<ActivateMethod, UpdateMethod>::back(const DeviceMatrix& weight_t_delta)
{
	//delta = f'(u) ** ((weight)^T * delta_output);
	//"**"は要素ごとの積
	UpdateMethod::getDelta(u, z, weight_t_delta, delta);
	
	//weightTDeltaを算出する
	//weightTDelta = weight^T * delta;
	getWeightTDelta();
	
	return weightTDelta;
}

//パラメータの更新
template <class ActivateMethod, class UpdateMethod>
void TwoLayerPerceptron<ActivateMethod, UpdateMethod>::update(const DeviceMatrix& x)
{
	updateMethod.update(x, delta, weight, bias);
}

}

