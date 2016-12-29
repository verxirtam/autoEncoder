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

#include "Backpropagation.h"

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
	std::vector<DeviceVector> vdv0;
	vdv0.push_back({11.0f, 12.0f});
	vdv0.push_back({21.0f, 22.0f, 23.0f});
	vdv0.push_back({31.0f, 32.0f, 33.0f, 34.0f});
	
	vdv0.resize(10);
	
	std::vector<DeviceVector> vdv;
	vdv = vdv0;
	
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

TEST_P(DeviceMatrixTest, CopyAssignmentOperator)
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
	
	DeviceMatrix dm1;
	dm1 = dm0;
	
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



TEST_P(DeviceMatrixTest, MoveConstructor)
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
	
	DeviceMatrix dm1(std::move(dm0));
	
	EXPECT_EQ(dm0.getRowCount(),    0);
	EXPECT_EQ(dm0.getColumnCount(), 0);
	EXPECT_EQ((dm0.getAddress() == nullptr), true);
	
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

TEST_P(DeviceMatrixTest, MoveAssignmentOperator)
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
	
	DeviceMatrix dm1;
	dm1 = std::move(dm0);
	
	EXPECT_EQ(dm0.getRowCount(),    0);
	EXPECT_EQ(dm0.getColumnCount(), 0);
	EXPECT_EQ((dm0.getAddress() == nullptr), true);
	
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

TEST(DeviceMatrixTest, useContainer)
{
	using host_matrix = std::vector<float>;
	std::vector<host_matrix> vhm;
	std::vector<unsigned int> vr;
	std::vector<unsigned int> vc;
	const unsigned int count = 100;
	for(unsigned int n = 0; n < count; n++)
	{
		host_matrix hm;
		
		unsigned int r = n + 1;
		unsigned int c = (n + 5) / 2;
		unsigned int imax = r * c;
		for(unsigned int i = 0; i < imax; i++)
		{
			hm.push_back(static_cast<float>(i));
		}
		vhm.push_back(hm);
		vr.push_back(r);
		vc.push_back(c);
	}
	
	std::vector<DeviceMatrix> vdm0;
	for(unsigned int n = 0; n < count; n++)
	{
		vdm0.push_back(DeviceMatrix(vr[n], vc[n], vhm[n]));
	}
	
	std::vector<DeviceMatrix> vdm;
	vdm = vdm0;
	vdm.resize(count * 3);
	unsigned int n;
	for(n = 0; n < count; n++)
	{
		DeviceMatrix dm = vdm[n];
		unsigned int r = vr[n];
		unsigned int c = vc[n];
		host_matrix hm;
		
		dm.get(hm);
		EXPECT_EQ(dm.getRowCount(),    r);
		EXPECT_EQ(dm.getColumnCount(), c);
		unsigned int imax = r * c;
		for(unsigned int i = 0; i < imax; i++)
		{
			EXPECT_EQ(hm[i], vhm[n][i]);
		}
	}
	for(; n < 3 * count; n++)
	{
		DeviceMatrix dm = vdm[n];
		EXPECT_EQ(dm.getRowCount(),    0);
		EXPECT_EQ(dm.getColumnCount(), 0);
		EXPECT_EQ((dm.getAddress() == nullptr), true);
	}
}


//////////////////////////////////////////////////////////////////////
// BackpropagationTest
//////////////////////////////////////////////////////////////////////

class BackpropagationTest : public ::testing::Test , public ::testing::WithParamInterface<unsigned int>
{
protected:
	void SetUp(){}
	void TearDown(){}
};

INSTANTIATE_TEST_CASE_P(InstantiateBackpropagationTest, BackpropagationTest, ::testing::Values(2, 3, 10, 100));

TEST_P(BackpropagationTest, Constructor)
{
	unsigned int layer_count = GetParam();
	Backpropagation b(layer_count);
}

TEST_P(BackpropagationTest, Init)
{
	unsigned int layer_count = GetParam();
	Backpropagation b(layer_count);
	
	std::vector<unsigned int> unit_count;
	for(unsigned int l = 0; l < layer_count; l++)
	{
		unsigned int uc = (l <= layer_count / 2) ? (layer_count - (l / 2)) : (layer_count / 2 + (l / 2));
		unit_count.push_back(uc);
	}
	
	b.init(unit_count);
	b.initRandom();
}

TEST(BackpropagationTest, Simple)
{
	Backpropagation b(3);
	//bの初期化
	b.init({1,1,1});
	std::vector<std::vector<float> > weight{{0.0f}, {2.0f}, {3.0f}};
	std::vector<std::vector<float> >   bias{{0.0f}, {1.0f}, {2.0f}};
	b.setWeight(weight);
	b.setBias(bias);
	
	//Forward
	////////////////////////////////////////
	//forwardの引数
	std::vector<float> x{1.0f};
	std::vector<float> y{0.0f};
	
	b.forward(x,y);
	//結果が想定通りか確認
	std::vector<float> d{0.0f};
	d[0] = std::tanh(2.0f * x[0] + 1.0);
	d[0] = std::tanh(3.0f * d[0] + 2.0);
	//結果が十分近いことを確認
	EXPECT_NEAR(y[0], d[0], 0.0001f);
	
	std::vector<std::vector<float> > u{{0.0f}, {0.0f}, {0.0f}};
	std::vector<std::vector<float> > z{{0.0f}, {0.0f}, {0.0f}};
	z[0][0] = x[0];
	u[1][0] = 2.0f * x[0]    + 1.0;
	z[1][0] = std::tanh(u[1][0]);
	u[2][0] = 3.0f * z[1][0] + 2.0;
	z[2][0] = std::tanh(u[2][0]);
	
	auto hz = b.getZAsVector();
	auto hu = b.getUAsVector();
	std::cout << "hz[1][0] = " << hz[1][0] << ", z[1][0] = " <<  z[1][0] << std::endl;
	std::cout << "hu[1][0] = " << hu[1][0] << ", u[1][0] = " <<  u[1][0] << std::endl;
	std::cout << "hz[2][0] = " << hz[2][0] << ", z[2][0] = " <<  z[2][0] << std::endl;
	std::cout << "hu[2][0] = " << hu[2][0] << ", u[2][0] = " <<  u[2][0] << std::endl;
	EXPECT_NEAR(hz[1][0], z[1][0], 0.0001f);
	EXPECT_NEAR(hu[1][0], u[1][0], 0.0001f);
	EXPECT_NEAR(hz[2][0], z[2][0], 0.0001f);
	EXPECT_NEAR(hu[2][0], u[2][0], 0.0001f);
	
	//Back
	////////////////////////////////////////
	
	b.back({1.0f});
	
	std::vector<std::vector<float> > delta = {{0.0f},{0.0f},{0.0f}};
	delta[2][0] = u[2][0] - 1.0f;
	float tanh_u = std::tanh(hu[1][0]);
	delta[1][0] = (1.0f - tanh_u * tanh_u) * delta[2][0] * weight[2][0];
	
	auto hdelta = b.getDeltaAsVector();
	std::cout << "hdelta[1][0] = " << hdelta[1][0] << ", delta[1][0] = " <<  delta[1][0] << std::endl;
	std::cout << "hdelta[2][0] = " << hdelta[2][0] << ", delta[2][0] = " <<  delta[2][0] << std::endl;
	EXPECT_NEAR(hdelta[1][0], delta[1][0], 0.0001f);
	EXPECT_NEAR(hdelta[2][0], delta[2][0], 0.0001f);
	
	std::vector<std::vector<float> > dEdW = {{0.0f},{0.0f},{0.0f}};
	dEdW[2][0] = delta[2][0] * z[1][0];
	dEdW[1][0] = delta[1][0] * z[0][0];

	auto hdEdW   = b.getDEDWAsVector();
	std::cout << "hdEdW[1][0] = " << hdEdW[1][0] << ", dEdW[1][0] = " <<  dEdW[1][0] << std::endl;
	std::cout << "hdEdW[2][0] = " << hdEdW[2][0] << ", dEdW[2][0] = " <<  dEdW[2][0] << std::endl;
}
TEST(BackpropagationTest, All)
{
	const unsigned int dimension = 1000;
	
	Backpropagation b(5);
	b.init({dimension, 500, 1, 500, dimension});
	b.initRandom();
	
	std::random_device rdev;
	std::mt19937 engine(rdev());
	std::uniform_real_distribution<float> urd(0.0f, 1.0f);
	
	int imax = 10;
	
	std::vector<float> r_[4];
	std::vector<float> r;
	
	std::cout << "r init start." << std::endl;
	
	#pragma omp parallel for
	for(int t = 0; t < 4; t++)
	{
		for(int i = 0; i < imax / 4; i++)
		{
			r_[t].push_back(urd(engine));
		}
	}
	
	r.reserve(imax);
	for(int t = 0; t < 4; t++)
	{
		r.insert(r.end(), r_[t].begin(), r_[t].end());
	}
	
	std::cout << "r init end." << std::endl;
	
	for(int i = 0; i < imax; i++)
	{
		
		std::vector<float> x(dimension, r[i]);
		std::vector<float> y;
		std::vector<float> d = x;
		b.forward(x, y);
		b.back(d);
		b.updateParameter();
	}
	std::vector<float> x(dimension, 0.5f);
	std::vector<float> y(dimension, 0.0f);
	b.forward(x, y);
	std::cout << "y = (" << y[0] << ", " << y[1] << ")" << std::endl;
}

//////////////////////////////////////////////////////////////////////
// main()
//////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	::testing::GTEST_FLAG(filter)="*Simple*";
	//::testing::GTEST_FLAG(filter)="*Input*:*Output*";
	
	
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

