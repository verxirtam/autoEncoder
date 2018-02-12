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
	float learning_rate = 0.5f ;
	float momentum      = 0.25f;
	
	DeviceMatrix       x(2, 1, {1.0f, 2.0f});
	DeviceMatrix   delta(2, 1, {1.0f, 2.0f});
	DeviceMatrix  weight(2, 2, {1.0f, 0.0f, 0.0f, 1.0f});
	DeviceVector    bias({1.0f, 2.0f});
	DeviceMatrix  weight_out(2, 2, {0.0f, 0.0f, 0.0f, 0.0f});
	DeviceVector    bias_out({0.0f, 0.0f});
	
	
	UpdateMethodMomentum u;
	
	u.init(dim_input, dim_output, minibatch_size);
	u.setLearningRate(learning_rate);
	u.setMomentum(momentum);
	
	u.update(x, delta, weight_out, bias_out);
	
	
	
	EXPECT_EQ(compare(weight, weight_out), 0.0f);
	EXPECT_EQ(compare(bias  , bias_out  ), 0.0f);
	
}




