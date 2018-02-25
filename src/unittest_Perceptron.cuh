#pragma once

#include "nn/Perceptron.cuh"
#include "nn/LayerInput.cuh"
#include "nn/LayerNull.cuh"

#include "nn/LayerInternal.cuh"
#include "nn/LayerOutputIdentity.cuh"

#include "nn/Func1to1ReLU.cuh"

class PerceptronTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

TEST(PerceptronTest, constructor)
{
	
	using namespace nn;
	
	using P = Perceptron<LayerInput, LayerNull, LayerNull>;
	
	P p;
}


TEST(PerceptronTest, init)
{
	using namespace nn;
	using P = Perceptron<LayerInput, LayerNull, LayerNull>;
	P p;
	unsigned int dim_input  = 2;
	unsigned int minibatch_size = 5;
	p.getInput().init(dim_input, minibatch_size);
	p.getInternal().init();
	p.getOutput().init();
}



TEST(PerceptronTest, simple)
{
	using namespace nn;
	using P = Perceptron<LayerInput, LayerInternal<Func1to1ReLU>, LayerOutputIdentity>;
	P p;
	
	p.getInput().init(1,1);
	p.getInternal().init(1,1,1);
	p.getOutput().init(1,1,1);
	
	p.getInternal().setWeight(DeviceMatrix(1, 1, {1.0f}));
	p.getInternal().setBias(DeviceVector({1.0f}));
	
	p.getOutput().setWeight(DeviceMatrix(1, 1, {1.0f}));
	p.getOutput().setBias(DeviceVector({1.0f}));
	
	//forwardの実行
	DeviceMatrix x(1, 1, {1.0f});
	DeviceMatrix y = p.forward(x);
	DeviceMatrix y_expect = DeviceMatrix(1, 1, {3.0f});
	//forwardの検証
	EXPECT_EQ(y.get(), y_expect.get());
	
	//backの実行
	//forwardの結果を引数とし、
	//パラメータが変化しないことを確認する
	//backの検証の場合はwtdが{0.0}となることを確認する
	DeviceMatrix wtd = p.back(DeviceMatrix(1, 1,{3.0f}));
	DeviceMatrix wtd_expect(1, 1, {0.0f});
	//backの検証
	EXPECT_EQ(wtd.get(), wtd_expect.get());
	
	//updateの実行
	//パラメータが変化しないことを確認する
	p.update();
	DeviceMatrix internal_weight = p.getInternal().getWeight();
	DeviceVector internal_bias   = p.getInternal().getBias();
	DeviceMatrix output_weight   = p.getOutput().getWeight();
	DeviceVector output_bias     = p.getOutput().getBias();
	
	DeviceMatrix internal_weight_expect(1, 1, {1.0f});
	DeviceVector internal_bias_expect({1.0f});
	DeviceMatrix output_weight_expect(1, 1, {1.0f});
	DeviceVector output_bias_expect({1.0f});
	
	//updateの検証
	EXPECT_EQ(internal_weight.get(), internal_weight_expect.get());
	EXPECT_EQ(internal_bias.get(), internal_bias_expect.get());
	EXPECT_EQ(output_weight.get(), output_weight_expect.get());
	EXPECT_EQ(output_bias.get(), output_bias_expect.get());
}



