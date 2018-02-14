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
	
	DeviceMatrix        u(2, 1, {-1.0f, 2.0f});
	DeviceMatrix        z(2, 1, { 0.0f, 2.0f});
	DeviceMatrix    z_out(2, 1, { 0.0f, 0.0f});
	DeviceMatrix z_expect(2, 1, { 0.0f, 2.0f});
	
	DeviceMatrix     weight_t_delta(2, 1, {2.0f, 3.0f});
	DeviceMatrix          delta_out(2, 1, {0.0f, 0.0f});
	DeviceMatrix       delta_expect(2, 1, {0.0f, 3.0f});
	
	
	//z = ActivateFunction(u)
	A::activate(u, z_out);
	
	//delta = Func(u, weight_t_delta)
	//      = ActivateFunction'(u) ** weight_t_delta
	A::getDelta(u, z, weight_t_delta, delta_out);
	
	EXPECT_EQ(compare(z_out    , z_expect    ), 0.0f);
	EXPECT_EQ(compare(delta_out, delta_expect), 0.0f);
	
	
}
