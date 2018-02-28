
#pragma once

#include "nn/Func1to1ReLU.cuh"
#include "nn/ActivateMethodOutputIdentity.cuh"

#include "cuda/DeviceVectorUtils.h"

class ActivateMethodOutputIdentityTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

TEST(ActivateMethodOutputIdentityTest, Simple)
{
	using namespace cuda;
	using namespace nn;
	
	using A = ActivateMethodOutputIdentity;
	
	DeviceMatrix        u(2, 1, {-1.0f, 2.0f});
	DeviceMatrix        z(2, 1, {-1.0f, 2.0f});
	DeviceMatrix    z_out(2, 1, { 0.0f, 0.0f});
	DeviceMatrix z_expect(2, 1, {-1.0f, 2.0f});
	
	DeviceMatrix            d(2, 1, { 2.0f,  3.0f});
	DeviceMatrix    delta_out(2, 1, { 0.0f,  0.0f});
	DeviceMatrix delta_expect(2, 1, {-3.0f, -1.0f});
	
	
	//z = ActivateFunction(u)
	A::activate(u, z_out);
	
	//delta = z - d
	A::getDelta(u, z, d, delta_out);
	
	EXPECT_EQ(compare(z_out    , z_expect    ), 0.0f);
	EXPECT_EQ(compare(delta_out, delta_expect), 0.0f);
	
	
}



