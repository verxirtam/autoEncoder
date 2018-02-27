#pragma once


#include "nn/Layer.h"
#include "nn/Func1to1ReLU.cuh"
#include "nn/ActivateMethodElementWise.cuh"
#include "nn/UpdateMethodMomentum.h"

#include "cuda/DeviceVectorUtils.h"


class LayerTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

TEST(LayerTest, Simple)
{
	using namespace nn;
	
	using L = Layer<ActivateMethodElementWise<Func1to1ReLU>, UpdateMethodMomentum>;
	L l;
	unsigned int dim_input      = 2;
	unsigned int dim_output     = 2;
	unsigned int minibatch_size = 1;
	
	DeviceMatrix  x(2, 1, {1.0f, 2.0f});
	DeviceMatrix  y(2, 1, {2.5f, 6.25f});
	DeviceMatrix  w(2, 2, {2.0f, 0.0f, 0.0f, 3.0f});
	DeviceVector  b({0.5f, 0.25f});
	
	l.init(dim_input, dim_output, minibatch_size);
	
	l.setWeight(w);
	l.setBias(b);
	
	const DeviceMatrix& y_out = l.forward(x);
	
	EXPECT_EQ(compare(y, y_out), 0.0f);
	
	l.back(x);
	EXPECT_EQ(1, 0);//TODO back()の結果を検証する処理を書くこと
	
	l.update(x);
	EXPECT_EQ(1, 0);//TODO update()の結果を検証する処理を書くこと
}




