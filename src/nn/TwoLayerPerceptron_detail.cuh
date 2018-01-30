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
template <class ActivateFunction>
void TwoLayerPerceptron<ActivateFunction>::init
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
	deltaWeight = DeviceMatrix(dim_output, dim_input);
	deltaBias   = DeviceVector(dim_output);
	
	//weightとbiasをランダムに初期化
	initWeightBias();
	
	//deltaWeightを0で初期化
	unsigned int N = deltaWeight.getRowCount();
	unsigned int M = deltaWeight.getColumnCount();
	deltaWeight.set(std::vector<float>(N * M, 0.0f));
	//deltaBiasを0で初期化
	N = deltaBias.getDimension();
	deltaBias.set(std::vector<float>(N, 0.0f));
}

//weight, biasをランダムに初期化する
template <class ActivateFunction>
void TwoLayerPerceptron<ActivateFunction>::initWeightBias(void)
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
template <class ActivateFunction>
const cuda::DeviceMatrix& TwoLayerPerceptron<ActivateFunction>::forward(const DeviceMatrix& x)
{
	//weight, biasを掛ける
	//u = weight * x + bias * _1B ^ T
	forwardLinear(x);
	
	//活性化関数を適用
	//z = ActivateFunction(u)
	nn::ActivateFunction<ActivateFunction>::activate(u, z);
	
	return z;
}

//順伝播の線型部分
//u = weight * x + bias * _1B ^ T
template <class ActivateFunction>
void TwoLayerPerceptron<ActivateFunction>::forwardLinear(const DeviceMatrix& x)
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


//逆伝播
template <class ActivateFunction>
const cuda::DeviceMatrix& TwoLayerPerceptron<ActivateFunction>::back(const DeviceMatrix& delta_output)
{
	return delta;
}

//パラメータの更新
template <class ActivateFunction>
void TwoLayerPerceptron<ActivateFunction>::update()
{
	
}

}

