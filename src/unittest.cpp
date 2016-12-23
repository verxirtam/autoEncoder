/*
 * =====================================================================================
 *
 *       Filename:  unittest.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月23日 18時39分38秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#define GTEST_LANG_CXX11 1



#include <gtest/gtest.h>

#include <tuple>

#include "DeviceVector.h"
#include "DeviceMatrix.h"

//////////////////////////////////////////////////////////////////////
// DeviceVectorTest
//////////////////////////////////////////////////////////////////////

class DeviceVectorTest : public ::testing::Test , public ::testing::WithParamInterface<unsigned int>
{
protected:
	void SetUp(){}
	void TearDown(){}
};

INSTANTIATE_TEST_CASE_P(InstantiateDeviceVectorTest, DeviceVectorTest, ::testing::Values(0, 10, 100, 1000, 10000));

TEST(DeviceVectorTest, DefaultConstructor)
{
	DeviceVector dv;
	EXPECT_EQ(dv.getDimension(), 0);
	EXPECT_EQ(dv.getAddress(), nullptr);
}

TEST(DeviceVectorTest, Constructor1)
{
	DeviceVector dv(3);
	EXPECT_EQ(dv.getDimension(), 3);
	EXPECT_EQ((dv.getAddress() != nullptr), true);
}

TEST(DeviceVectorTest, Constructor2)
{
	DeviceVector dv(std::vector<float>{1.0f, 2.0f});
	EXPECT_EQ(dv.getDimension(), 2);
	EXPECT_EQ((dv.getAddress() != nullptr), true);
	
	std::vector<float> hv(2, 0.0f);
	dv.get(hv.data());
	EXPECT_EQ(hv[0], 1.0f);
	EXPECT_EQ(hv[1], 2.0f);
}

TEST(DeviceVectorTest, Constructor3)
{
	DeviceVector dv{1.0f, 2.0f};
	EXPECT_EQ(dv.getDimension(), 2);
	EXPECT_EQ((dv.getAddress() != nullptr), true);
	
	std::vector<float> hv;
	dv.get(hv);
	EXPECT_EQ(hv[0], 1.0f);
	EXPECT_EQ(hv[1], 2.0f);
}

TEST(DeviceVectorTest, CopyConstructor)
{
	using namespace std;
	
	DeviceVector dv0(3);
	vector<float> h_dv0{1.0f, 2.0f, 3.0f};
	vector<float> h_dv1{0.0f, 0.0f, 0.0f};
	dv0.set(h_dv0.data());
	DeviceVector dv1(dv0);
	dv1.get(h_dv1.data());
	
	EXPECT_EQ(dv1.getDimension(), 3);
	EXPECT_EQ(h_dv1[0], 1.0f);
	EXPECT_EQ(h_dv1[1], 2.0f);
	EXPECT_EQ(h_dv1[2], 3.0f);
}

TEST(DeviceVectorTest, CopyAssignmentOperator)
{
	using namespace std;
	
	DeviceVector dv0(3);
	vector<float> h_dv0{1.0f, 2.0f, 3.0f};
	vector<float> h_dv1{0.0f, 0.0f, 0.0f};
	dv0.set(h_dv0.data());
	
	DeviceVector dv1;
	dv1 = dv0;
	dv1.get(h_dv1.data());
	
	EXPECT_EQ(dv1.getDimension(), 3);
	EXPECT_EQ(h_dv1[0], 1.0f);
	EXPECT_EQ(h_dv1[1], 2.0f);
	EXPECT_EQ(h_dv1[2], 3.0f);
}

TEST(DeviceVectorTest, MoveConstructor1)
{
	using namespace std;
	
	
	DeviceVector dv1(DeviceVector{1.0f, 2.0f, 3.0f});
	
	vector<float> h_dv1{0.0f, 0.0f, 0.0f};
	dv1.get(h_dv1.data());
	
	EXPECT_EQ(dv1.getDimension(), 3);
	EXPECT_EQ(h_dv1[0], 1.0f);
	EXPECT_EQ(h_dv1[1], 2.0f);
	EXPECT_EQ(h_dv1[2], 3.0f);
}

TEST(DeviceVectorTest, MoveConstructor2)
{
	using namespace std;
	
	DeviceVector dv0(DeviceVector{1.0f, 2.0f, 3.0f});
	DeviceVector dv1(std::move(dv0));
	
	vector<float> h_dv1{0.0f, 0.0f, 0.0f};
	dv1.get(h_dv1.data());
	
	EXPECT_EQ(dv1.getDimension(), 3);
	EXPECT_EQ(h_dv1[0], 1.0f);
	EXPECT_EQ(h_dv1[1], 2.0f);
	EXPECT_EQ(h_dv1[2], 3.0f);
	
	EXPECT_EQ(dv0.getDimension(), 0);
	EXPECT_EQ(dv0.getAddress(), nullptr);
}
TEST(DeviceVectorTest, MoveAssignmentOperator1)
{
	using namespace std;
	
	
	DeviceVector dv1;
	dv1 = DeviceVector{1.0f, 2.0f, 3.0f};
	
	vector<float> h_dv1;
	dv1.get(h_dv1);
	
	EXPECT_EQ(dv1.getDimension(), 3);
	EXPECT_EQ(h_dv1[0], 1.0f);
	EXPECT_EQ(h_dv1[1], 2.0f);
	EXPECT_EQ(h_dv1[2], 3.0f);
}

TEST(DeviceVectorTest, MoveAssignmentOperator2)
{
	using namespace std;
	
	DeviceVector dv0(DeviceVector{1.0f, 2.0f, 3.0f});
	
	DeviceVector dv1;
	dv1 = std::move(dv0);
	
	vector<float> h_dv1;
	dv1.get(h_dv1);
	
	EXPECT_EQ(dv1.getDimension(), 3);
	EXPECT_EQ(h_dv1[0], 1.0f);
	EXPECT_EQ(h_dv1[1], 2.0f);
	EXPECT_EQ(h_dv1[2], 3.0f);
	
	EXPECT_EQ(dv0.getDimension(), 0);
	EXPECT_EQ(dv0.getAddress(), nullptr);
}

TEST_P(DeviceVectorTest, set)
{
	unsigned int dimension = GetParam();
	DeviceVector dv(dimension);
	std::vector<float> hv;
	//下記の形のベクトルを設定する
	//{1.0f, 2.0f, ...}
	for(unsigned int i = 0; i < dimension; i++)
	{
		hv.push_back(static_cast<float>(i));
	}
	
	dv.set(hv);
	
	std::vector<float> result;
	
	dv.get(result);
	
	for(unsigned int i = 0; i < dimension; i++)
	{
		EXPECT_EQ(result[i], hv[i]);
	}
}

TEST(DeviceVectorTest, useContainer)
{
	std::vector<DeviceVector> vdv;
	vdv.push_back({11.0f, 12.0f});
	vdv.push_back({21.0f, 22.0f, 23.0f});
	vdv.push_back({31.0f, 32.0f, 33.0f, 34.0f});
	
	vdv.resize(10);
	
	std::vector<float> result;
	
	EXPECT_EQ(vdv[0].getDimension(),2);
	vdv[0].get(result);
	EXPECT_EQ(result.size(), 2);
	EXPECT_EQ(result[0], 11.0f);
	EXPECT_EQ(result[1], 12.0f);
	
	EXPECT_EQ(vdv[1].getDimension(),3);
	vdv[1].get(result);
	EXPECT_EQ(result.size(), 3);
	EXPECT_EQ(result[0], 21.0f);
	EXPECT_EQ(result[1], 22.0f);
	EXPECT_EQ(result[2], 23.0f);
	
	EXPECT_EQ(vdv[2].getDimension(),4);
	vdv[2].get(result);
	EXPECT_EQ(result.size(), 4);
	EXPECT_EQ(result[0], 31.0f);
	EXPECT_EQ(result[1], 32.0f);
	EXPECT_EQ(result[2], 33.0f);
	EXPECT_EQ(result[3], 34.0f);
	
	EXPECT_EQ(vdv[3].getDimension(),0);
	EXPECT_EQ((vdv[3].getAddress()==nullptr), true);
	EXPECT_EQ(vdv[4].getDimension(),0);
	EXPECT_EQ((vdv[4].getAddress()==nullptr), true);
	EXPECT_EQ(vdv[5].getDimension(),0);
	EXPECT_EQ((vdv[5].getAddress()==nullptr), true);
	EXPECT_EQ(vdv[6].getDimension(),0);
	EXPECT_EQ((vdv[6].getAddress()==nullptr), true);
	EXPECT_EQ(vdv[7].getDimension(),0);
	EXPECT_EQ((vdv[7].getAddress()==nullptr), true);
	EXPECT_EQ(vdv[8].getDimension(),0);
	EXPECT_EQ((vdv[8].getAddress()==nullptr), true);
	EXPECT_EQ(vdv[9].getDimension(),0);
	EXPECT_EQ((vdv[9].getAddress()==nullptr), true);
}

//////////////////////////////////////////////////////////////////////
// DeviceMatrixTest
//////////////////////////////////////////////////////////////////////

using RowColumn = std::tuple<unsigned int, unsigned int>;
class DeviceMatrixTest : public ::testing::Test , public ::testing::WithParamInterface<RowColumn>
{
protected:
	void SetUp(){}
	void TearDown(){}
};

std::vector<unsigned int> count{0, 1, 10, 100, 1000};
INSTANTIATE_TEST_CASE_P
	(
		InstantiateDeviceMatrixTest,
		DeviceMatrixTest,
		::testing::Combine
			(
				::testing::ValuesIn(count),
				::testing::ValuesIn(count)
			)
	);

TEST(DeviceMatrixTest, DefaultConstructor)
{
	DeviceMatrix dm;
	EXPECT_EQ(dm.getRowCount()   , 0);
	EXPECT_EQ(dm.getColumnCount(), 0);
	EXPECT_EQ(dm.getAddress(), nullptr);
}

TEST_P(DeviceMatrixTest, Constructor1)
{
	unsigned int r = std::get<0>(GetParam());
	unsigned int c = std::get<1>(GetParam());
	DeviceMatrix dm(r, c);
	EXPECT_EQ(dm.getRowCount(),    r);
	EXPECT_EQ(dm.getColumnCount(), c);
	if(r * c != 0)
	{
		EXPECT_EQ((dm.getAddress() != nullptr), true);
	}
	else
	{
		EXPECT_EQ((dm.getAddress() == nullptr), true);
	}
}

TEST_P(DeviceMatrixTest, Constructor2)
{
	//行列のサイズの取得
	unsigned int r = std::get<0>(GetParam());
	unsigned int c = std::get<1>(GetParam());
	//初期化用のデータの設定
	std::vector<float> hm;
	for(unsigned int j = 0; j < c; j++)
	{
		for(unsigned int i = 0; i < r; i++)
		{
			float value = static_cast<float>(i + 1) + 1.0f / static_cast<float>(j + 1);
			hm.push_back(value);
		}
	}
	
	//コンストラクタの実行
	DeviceMatrix dm(r, c, hm);
	//初期化の内容のチェック
	EXPECT_EQ(dm.getRowCount(),    r);
	EXPECT_EQ(dm.getColumnCount(), c);
	if(r * c != 0)
	{
		EXPECT_EQ((dm.getAddress() != nullptr), true);
	}
	else
	{
		EXPECT_EQ((dm.getAddress() == nullptr), true);
	}
	
	//初期化した値のチェック
	std::vector<float> result;
	dm.get(result);
	
	for(unsigned int j = 0; j < c; j++)
	{
		for(unsigned int i = 0; i < r; i++)
		{
			EXPECT_EQ(result[i + j * r], hm[i + j * r]);
		}
	}
}
TEST(DeviceMatrixTest,Constructor3)
{
	unsigned int r = 2;
	unsigned int c = 3;
	DeviceMatrix dm(r, c, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
	EXPECT_EQ(dm.getRowCount(),    r);
	EXPECT_EQ(dm.getColumnCount(), c);
	
	std::vector<float> result;
	
	dm.get(result);
	
	for(unsigned int j = 0; j < c; j++)
	{
		for(unsigned int i = 0; i < r; i++)
		{
			unsigned int n = i + j * r;
			EXPECT_EQ(result[n], static_cast<float>(n + 1));
		}
	}
}

TEST_P(DeviceMatrixTest, CopyConstructor)
{
	unsigned int r = std::get<0>(GetParam());
	unsigned int c = std::get<1>(GetParam());
	DeviceMatrix dm0(r, c);
	//初期化用のデータの設定
	std::vector<float> hm;
	for(unsigned int j = 0; j < c; j++)
	{
		for(unsigned int i = 0; i < r; i++)
		{
			float value = static_cast<float>(i + 1) + 1.0f / static_cast<float>(j + 1);
			hm.push_back(value);
		}
	}
	dm0.set(hm);
	
	DeviceMatrix dm1(dm0);
	
	EXPECT_EQ(dm1.getRowCount(),    r);
	EXPECT_EQ(dm1.getColumnCount(), c);
	if(r * c != 0)
	{
		EXPECT_EQ((dm1.getAddress() != nullptr), true);
	}
	else
	{
		EXPECT_EQ((dm1.getAddress() == nullptr), true);
	}
	
	//初期化した値のチェック
	std::vector<float> result;
	dm1.get(result);
	
	for(unsigned int j = 0; j < c; j++)
	{
		for(unsigned int i = 0; i < r; i++)
		{
			EXPECT_EQ(result[i + j * r], hm[i + j * r]);
		}
	}
}


int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

