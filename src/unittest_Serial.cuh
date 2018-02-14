
#include "nn/Serial.h"
#include "nn/LayerNull.cuh"

#include "cuda/DeviceMatrix.h"

class SerialTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

TEST(SerialTest, Simple)
{
	using namespace nn;
	using L = LayerNull;
	using S = Serial<L, L, L>;
	S s;
	s.getMember<0>().init();
	s.getMember<1>().init();
	s.getMember<2>().init();

	auto x = cuda::DeviceMatrix(1,1,std::vector<float>(1, 0.0f));
	auto d = x;
	s.forward(x);
	s.back(d);
	s.update(x);
}



TEST(SerialTest, Iteration)
{
	using namespace nn;
	using L0 = LayerNull;
	using L1 = Serial<L0, L0, L0>;
	using L2 = Serial<L1, L1, L1>;
	using S = L2;
	S s;
	s.getMember<0>().getMember<0>().init();
	s.getMember<0>().getMember<1>().init();
	s.getMember<0>().getMember<2>().init();
	s.getMember<1>().getMember<0>().init();
	s.getMember<1>().getMember<1>().init();
	s.getMember<1>().getMember<2>().init();
	s.getMember<2>().getMember<0>().init();
	s.getMember<2>().getMember<1>().init();
	s.getMember<2>().getMember<2>().init();
	
	auto x = cuda::DeviceMatrix(1,1,std::vector<float>(1, 0.0f));
	auto d = x;
	
	s.forward(x);
	s.back(d);
	s.update(x);
}

