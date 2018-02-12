#pragma once

#include "nn/Func1to1ReLU.cuh"
#include "nn/ActivateMethodElementWise.cuh"

#include "cuda/DeviceVectorUtils.h"

class ActivateMethodElementWiseTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

TEST(ActivateMethodElementWiseTest, Simple)
{
	using namespace cuda;
	using namespace nn;
	
	using A = ActivateMethodElementWise<Func1to1ReLU>;
	
	DeviceMatrix     u(2, 1, {1.0f, 2.0f});
	DeviceMatrix     z(2, 1, {0.0f, 0.0f});
	DeviceMatrix z_out(2, 1, {0.0f, 0.0f});
	
	DeviceMatrix     weight_t_delta(2, 2, {1.0f, 0.0f, 0.0f, 1.0f});
	DeviceMatrix              delta(2, 2, {1.0f, 0.0f, 0.0f, 1.0f});
	DeviceMatrix          delta_out(2, 2, {0.0f, 0.0f, 0.0f, 0.0f});
	
	
	A::activate(u, z_out);
	A::getDelta(u, z, weight_t_delta, delta);
	
	EXPECT_EQ(compare(z    , z_out    ), 0.0f);
	EXPECT_EQ(compare(delta, delta_out), 0.0f);
	
	
}
