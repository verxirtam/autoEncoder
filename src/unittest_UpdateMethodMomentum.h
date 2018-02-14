#pragma once


#include "nn/UpdateMethodMomentum.h"

#include "cuda/DeviceVectorUtils.h"


class UpdateMethodMomentumTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

TEST(UpdateMethodMomentumTest, Simple)
{
	using namespace cuda;
	using namespace nn;
	
	unsigned int dim_input      = 2;
	unsigned int dim_output     = 2;
	unsigned int minibatch_size = 1;
	
	//weightの更新
	//-----------------------------------------------
	// deltaWeight = - (learningRate / B) * delta * x^T + momentum * deltaWeight
	// weight      = weight + deltaWeight
	//biasの更新
	//-----------------------------------------------
	// deltaBias = - (learningRate / B) * delta * _1B + momentum * deltaBias
	// bias      = bias + deltaBias
	
	float learning_rate = 0.5f ;
	float momentum      = 0.25f;
	DeviceMatrix            x(2, 1, {1.0f, 2.0f});
	DeviceMatrix        delta(2, 1, {2.0f, 4.0f});
	DeviceMatrix       weight(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
	DeviceMatrix delta_weight(2, 2, {4.0f, 0.0f, 0.0f, 8.0f});
	DeviceVector         bias({1.0f, 2.0f});
	DeviceVector   delta_bias({4.0f, 8.0f});
	
	DeviceMatrix delta_weight_expect(2, 2, {0.0f, -2.0f, -2.0f, -2.0f});
	DeviceMatrix       weight_expect(2, 2, {1.0f,  0.0f,  1.0f,  2.0f});
	DeviceVector   delta_bias_expect({0.0f, 0.0f});
	DeviceVector         bias_expect({1.0f, 2.0f});
	
	
	UpdateMethodMomentum u;
	
	u.init(dim_input, dim_output, minibatch_size);
	u.setLearningRate(learning_rate);
	u.setMomentum(momentum);
	u.setDeltaWeight(delta_weight);
	u.setDeltaBias(delta_bias);
	
	u.update(x, delta, weight, bias);
	
	const DeviceMatrix& delta_weight_out = u.getDeltaWeight();
	const DeviceVector& delta_bias_out   = u.getDeltaBias();
	
	
	EXPECT_EQ(compare(delta_weight_expect, delta_weight_out), 0.0f);
	EXPECT_EQ(compare(      weight_expect,           weight), 0.0f);
	
	EXPECT_EQ(compare(delta_bias_expect, delta_bias_out), 0.0f);
	EXPECT_EQ(compare(      bias_expect,           bias), 0.0f);
	
}




