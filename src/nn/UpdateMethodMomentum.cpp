/*
 * =====================================================================================
 *
 *       Filename:  UpdateMethodMomentum.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年02月05日 01時52分36秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "UpdateMethodMomentum.h"

namespace nn
{


//初期化
void UpdateMethodMomentum::init(unsigned int dim_input, unsigned int dim_output, unsigned int minibatch_size)
{
	deltaWeight = DeviceMatrix(dim_output, dim_input);
	deltaBias   = DeviceVector(dim_output);
	_1B = DeviceVector::get1Vector(minibatch_size);
	
	//deltaWeightを0で初期化
	unsigned int N = deltaWeight.getRowCount();
	unsigned int M = deltaWeight.getColumnCount();
	deltaWeight.set(std::vector<float>(N * M, 0.0f));
	
	//deltaBiasを0で初期化
	N = deltaBias.getDimension();
	deltaBias.set(std::vector<float>(N, 0.0f));
	
}
//パラメータの更新
void UpdateMethodMomentum::update(const DeviceMatrix& x, const DeviceMatrix& delta, DeviceMatrix& weight, DeviceVector& bias)
{
	
	//weightの更新
	//-----------------------------------------------
	// deltaWeight = momentum * deltaWeight - learningRate * dEdW
	// weight      = weight + deltaWeight
	//-----------------------------------------------
	// deltaWeightの算出
	// deltaWeight = momentum * deltaWeight - learningRate * dEdW
	//             = momentum * deltaWeight - learningRate * (1 / B) * delta * x^T
	//             = - (learningRate / B) * delta * x^T + momentum * deltaWeight
	unsigned int minibatch_size = _1B.getDimension();
	float alpha = - learningRate / minibatch_size;
	float beta  = momentum;
	cuda::Sgemm(&alpha, CUBLAS_OP_N, delta, CUBLAS_OP_T, x, &beta, deltaWeight);
	// weightの更新
	// weight = weight + deltaWeight
	//        = 1.0f * weight + 1.0f * deltaWeight
	alpha = 1.0f;
	beta  = 1.0f;
	cuda::Sgeam(&alpha, CUBLAS_OP_N, weight, &beta, CUBLAS_OP_N, deltaWeight, weight);
	
	
	
	//biasの更新
	//-----------------------------------------------
	// deltaBias = momentum * deltaBias - learningRate * dEdb
	// bias      = bias + deltaBias
	//-----------------------------------------------
	// deltaBiasの算出
	// deltaBias = momentum * deltaBias - learningRate * dEdb
	//           = momentum * deltaBias - learningRate * ((1.0f / B) * delta * _1B)
	//           = momentum * deltaBias - (learningRate / B) * delta * _1B
	//           = - (learningRate / B) * delta * _1B + momentum * deltaBias
	alpha = - learningRate / minibatch_size;
	beta  = momentum;
	cuda::Sgemv(&alpha, CUBLAS_OP_N, delta, _1B, &beta, deltaBias);
	// biasの更新
	// bias = bias + deltaBias
	//      = 1.0f * deltaBias + bias
	alpha = 1.0f;
	cuda::Saxpy(&alpha, deltaBias, bias);
}

}
