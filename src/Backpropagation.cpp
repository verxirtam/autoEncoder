/*
 * =====================================================================================
 *
 *       Filename:  Backpropagation.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月19日 03時55分38秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "Backpropagation.h"

void Backpropagation::init(const std::vector<unsigned int>& unit_count)
{
	if(layerCount != unit_count.size())
	{
		std::cout << "error at Backpropagation::init() : layerCount != unit_count.size()." << std::endl;
	}
	unitCount = unit_count;

	u.clear();
	z.clear();
	weight.clear();
	bias.clear();
	dEdW.clear();
	dEdb.clear();

	for(unsigned int l = 0; l < layerCount; l++)
	{
		u.push_back(upDeviceVector(new DeviceVector(unitCount[l])));
		z.push_back(upDeviceVector(new DeviceVector(unitCount[l])));
		if(l == 0)
		{
			//インデックスl = 0は使用しないのでダミーの値を格納する。
			weight.push_back(upDeviceMatrix(new DeviceMatrix(1,1)));
			bias.push_back(  upDeviceVector(new DeviceVector(1  )));
			dEdW.push_back(  upDeviceMatrix(new DeviceMatrix(1,1)));
			dEdb.push_back(  upDeviceVector(new DeviceVector(1  )));
		}
		else
		{
			weight.push_back(upDeviceMatrix(new DeviceMatrix(unitCount[l],unitCount[l-1])));
			bias.push_back(  upDeviceVector(new DeviceVector(unitCount[l])));
			dEdW.push_back(  upDeviceMatrix(new DeviceMatrix(unitCount[l],unitCount[l-1])));
			dEdb.push_back(  upDeviceVector(new DeviceVector(unitCount[l])));
		}
	}
}

void Backpropagation::initRandom(void)
{
	std::random_device rdev;
	std::mt19937 engine(rdev());
	std::uniform_real_distribution<float> urd(0.0f, 1.0f);

	for(auto&& w : weight)
	{
		unsigned int M = w->getRowCount();
		unsigned int N = w->getColumnCount();
		std::vector<float> h_w;
		for(unsigned int i =0; i < M * N; i++)
		{
			h_w.push_back(urd(engine));
		}
		w->set(h_w.data());
	}
	for(auto&& b : bias)
	{
		unsigned int N = b->getDimension();
		std::vector<float> h_b;
		for(unsigned int i =0; i < N; i++)
		{
			h_b.push_back(urd(engine));
		}
		b->set(h_b.data());
	}
}

