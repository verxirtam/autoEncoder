#pragma once

#include "nn/BackpropagationUtils.cuh"
#include "nn/LayerInternal.cuh"
#include "nn/LayerOutputIdentity.cuh"

class PerceptronUtilsTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

TEST(PerceptronUtilsTest, simple)
{
	
	using namespace nn;
	using P = Perceptron<LayerInput, LayerInternal<Func1to1ReLU>, LayerOutputIdentity>;
	
	P p;
	p.getInput()   .init(1, 1);
	p.getInternal().init(1, 1, 1);
	p.getOutput()  .init(1, 1, 1);
	
	p.getInternal().setWeight(DeviceMatrix(1, 1, {1.0f}));
	p.getInternal().setBias(DeviceVector({2.0f}));
	p.getOutput()  .setWeight(DeviceMatrix(1, 1, {3.0f}));
	p.getOutput()  .setBias(DeviceVector({4.0f}));
	
	
	auto&& pv = getParameterVector(p);
	
	EXPECT_EQ(pv, std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
}




