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

#include <algorithm>
#include <numeric>
#include <limits>
#include <fstream>

#include "DeviceVector.h"
#include "DeviceMatrix.h"

#include "Backpropagation.h"

#include "CuRandFunction.h"

#include "CuSolverDnManager.h"
#include "CuSolverDnFunction.h"

#include "Normalization.h"

#include "AutoEncoder.h"

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
	d[0] = 3.0f * d[0] + 2.0;//最後のレイヤの活性化関数は恒等写像
	//結果が十分近いことを確認
	EXPECT_NEAR(y[0], d[0], 0.0001f);
	
	std::vector<std::vector<float> > u{{0.0f}, {0.0f}, {0.0f}};
	std::vector<std::vector<float> > z{{0.0f}, {0.0f}, {0.0f}};
	z[0][0] = x[0];
	u[1][0] = 2.0f * x[0]    + 1.0;
	z[1][0] = std::tanh(u[1][0]);
	u[2][0] = 3.0f * z[1][0] + 2.0;
	z[2][0] = u[2][0];//最後のレイヤは恒等写像
	
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

/////////////////////////////////////////////////////////////////////////////////
class BackpropagationAllTest : public ::testing::Test , public ::testing::WithParamInterface<unsigned int>
{
protected:
	void SetUp(){}
	void TearDown(){}
};

INSTANTIATE_TEST_CASE_P
	(
		InstantiateBackpropagationAllTest,
		BackpropagationAllTest,
		//::testing::Values(205)
		//::testing::Values(200, 201, 202, 203, 204, 205, 206)
		//::testing::Values(200, 206, 213, 219, 225)
		//::testing::Values(200, 225, 250, 275, 300)
		//::testing::Values(10, 50, 100, 200, 300, 400, 500, 625, 750, 875, 1000, 1024)
		//::testing::Values(10, 1024, 1025, 2000)
		::testing::Values(10, 1025)
		//::testing::Values(500, 625, 750, 875, 1000)
	);

void BackpropagationTest_All_showInfo
	(
		unsigned int dimension,
		int n,
		const Backpropagation& b,
		const std::vector<float>& x,
		const std::vector<float>& y
	)
{
	std::cout << "//" << " dimension = " << dimension << std::endl;
	std::cout << "//" << " n = " << n << std::endl;
	std::cout << "///////////////////////////////////////////////////////" << std::endl;
	std::cout << "x = (";
	unsigned int imax = std::min(x.size(), 10ul);
	for(unsigned int i = 0; i < imax; i++)
	{
		std::cout << x[i] << ", ";
	}
	std::cout << "...)" << std::endl;

	auto z = b.getZAsVector();
	std::cout << "z = (";
	for(auto&& z_l : z)
	{
		unsigned int imax = std::min(z_l.size(), 10ul);
		for(unsigned int i = 0; i < imax; i++)
		{
			std::cout << z_l[i] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << "...)" << std::endl;

	auto u = b.getUAsVector();
	std::cout << "u = (";
	for(auto&& u_l : u)
	{
		unsigned int imax = std::min(u_l.size(), 10ul);
		for(unsigned int i = 0; i < imax; i++)
		{
			std::cout << u_l[i] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << "...)" << std::endl;

	auto dEdW = b.getDEDWAsVector();
	std::cout << "dEdW = (";
	for(auto&& w_l : dEdW)
	{
		unsigned int imax = std::min(w_l.size(), 10ul);
		for(unsigned int i = 0; i < imax; i++)
		{
			std::cout << w_l[i] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << "...)" << std::endl;

	auto delta = b.getDeltaAsVector();
	std::cout << "delta = (";
	for(auto&& d_l : delta)
	{
		unsigned int imax = std::min(d_l.size(), 10ul);
		for(unsigned int i = 0; i < imax; i++)
		{
			std::cout << d_l[i] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << "...)" << std::endl;

	auto weight = b.getWeightAsVector();
	std::cout << "weight = (";
	for(auto&& w_l : weight)
	{
		unsigned int imax = std::min(w_l.size(), 10ul);
		for(unsigned int i = 0; i < imax; i++)
		{
			std::cout << w_l[i] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << "...)" << std::endl;

	auto bias = b.getBiasAsVector();
	std::cout << "bias = (";
	for(auto&& b_l : bias)
	{
		unsigned int imax = std::min(b_l.size(), 10ul);
		for(unsigned int i = 0; i < imax; i++)
		{
			std::cout << b_l[i] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << "...)" << std::endl;


	std::cout << "y = (";
	imax = std::min(y.size(), 10ul);
	for(unsigned int i = 0; i < imax; i++)
	{
		std::cout << y[i] << ", ";
	}
	std::cout << "...)" << std::endl;

}


TEST_P(BackpropagationAllTest, All)
{
	const unsigned int dimension = GetParam();
	
	std::vector<unsigned int> unit_count{dimension, dimension / 2, 1, dimension / 2, dimension};
	Backpropagation b(unit_count.size());
	b.init(unit_count);
	
	std::random_device rdev;
	std::mt19937 engine(rdev());
	std::uniform_real_distribution<float> urd(0.0f, 1.0f);
	
	int nmax = 100;
	
	std::vector<float> r;
	
	std::cout << "r init start." << std::endl;
	
	for(int n = 0; n < nmax; n++)
	{
		r.push_back(urd(engine));
	}
	
	std::cout << "r init end." << std::endl;
	
	BackpropagationTest_All_showInfo(dimension, -1, b, {}, {});
	
	for(int n = 0; n < nmax; n++)
	{
		std::cout << "n =  " << n << std::endl;
		std::vector<float> x(dimension, r[n]);
		std::vector<float> y;
		std::vector<float> d = x;
		b.forward(x, y);
		b.back(d);
		b.updateParameter();
		
		if(n == nmax -1)
		{
			BackpropagationTest_All_showInfo(dimension, n, b, x, y);
		}
	}
	std::vector<float> x(dimension, 0.5f);
	std::vector<float> y(dimension, 0.0f);
	b.forward(x, y);
	std::cout << "y = (" << y[0] << ", " << y[1] << ")" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////
class BackpropagationFunctionTest :
	public ::testing::Test ,
	public ::testing::WithParamInterface<std::tuple<unsigned int, unsigned int>>
{
protected:
	void SetUp(){}
	void TearDown(){}
};
INSTANTIATE_TEST_CASE_P
	(
		InstantiateBackpropagationFunctionTest,
		BackpropagationFunctionTest,
		::testing::Combine
			(
				::testing::ValuesIn(std::vector<unsigned int>{1u, 2u, 10u, 100u, 1025u, 2050u}),
				::testing::ValuesIn(std::vector<unsigned int>{1u, 2u, 10u})
			)
	);
//std::vector<float>同士の差が許容誤差未満であることを確認する
//その差の絶対値の最大値を求める
float compareVector(const std::vector<float>& x,const std::vector<float>& y, float error = 0.0625)
{
	if(x.size() != y.size())
	{
		std::cout << "x.size() = " << x.size() << ", y.size() = " << y.size() << std::endl;
		throw std::runtime_error("x.size() != y.size()");
	}
	float diff = std::inner_product
		(
		 x.begin(), x.end(), y.begin(), 0.0f,
		 [](float _x, float _y){return std::max(_x, _y);},
		 [](float _x, float _y){return std::abs(_x - _y);}
		);
	EXPECT_NEAR(diff, 0.0f, error);
	return diff;
}
//CUDAの算出結果とhostでの算出結果と一致することを確認する
TEST_P(BackpropagationFunctionTest, Kernel)
{
	unsigned int d0 = std::get<0>(GetParam());
	unsigned int d1 = std::get<1>(GetParam());
	Backpropagation b(3);
	b.init({d0, d1, d0});
	
	std::vector<float> x(d0, 0.5f);
	std::vector<float> y;
	auto d = x;

	b.forward(x, y);
	b.back(d);
	
	auto du = b.getUAsVector();
	auto dz = b.getZAsVector();
	auto dweight = b.getWeightAsVector();
	auto hz = du;
	
	hz[0] = x;
	
	unsigned int lmax = du.size();
	//z = f(u);
	for(unsigned int l = 1; l < lmax; l++)
	{
		std::transform(hz[l].begin(), hz[l].end(), hz[l].begin(), [](float _x){return std::tanh(_x);});
		float max_error = compareVector(hz[l], dz[l]);
		std::cout << "max_error(z[" << l << "]) = " << max_error << std::endl;
	}
	
	auto hu = b.getUAsVector();
	auto dbias = b.getBiasAsVector();
	//u = weight * z + bias;
	for(unsigned int l = 1; l < lmax; l++)
	{
		unsigned int imax = hu[l].size();
		for(unsigned int i = 0; i < imax; i++)
		{
			hu[l][i] = 0.0f;
			unsigned int kmax = hz[l -1].size();
			for(unsigned int k = 0; k < kmax; k++)
			{
				hu[l][i] += dweight[l][i + k * imax] * dz[l - 1][k];
			}
			hu[l][i] += dbias[l][i];
		}
		float max_error = compareVector(hu[l], du[l]);
		std::cout << "max_error(u[" << l << "]) = " << max_error << std::endl;
	}
	
	auto ddelta = b.getDeltaAsVector();
	auto hdelta = ddelta;
	//hdelta[lmax -1] = hdelta[lmax -1] - d;
	std::transform
		(
			du[lmax -1].begin(), du[lmax -1].end(), d.begin(), hdelta[lmax -1].begin(),
			[](float _x, float _y){return _x - _y;}
		);
	float max_error = compareVector(hdelta[lmax - 1], ddelta[lmax - 1]);
	std::cout << "max_error(delta[" << (lmax - 1) << "]) = " << max_error << std::endl;
	
	//hdelta[l] = f'(u[l]) ** (weight[l + 1]^T * delta[l + 1]);
	for(unsigned int l = lmax - 2; l >= 1; l--)
	{
		unsigned int uc_l = du[l].size();
		unsigned int kmax = du[l + 1].size();
		for(unsigned int j = 0; j < uc_l; j++)
		{
			float hwtdelta_lj = 0.0f;
			for(unsigned int k = 0; k < kmax; k++)
			{
				hwtdelta_lj += dweight[l + 1][k + j * kmax] * hdelta[l + 1][k];
			}
			float tanh_u_lj = std::tanh(du[l][j]);
			float fd_u_lj = (1.0f - tanh_u_lj * tanh_u_lj);
			hdelta[l][j] = fd_u_lj * hwtdelta_lj;
		}
		float max_error = compareVector(hdelta[l], ddelta[l]);
		std::cout << "max_error(delta[" << l << "]) = " << max_error << std::endl;
	}
	auto ddedw = b.getDEDWAsVector();
	auto hdedw = ddedw;
	//dEdW[l] = delta[l] * z[l-1]^T
	for(unsigned int l = 1; l < lmax; l++)
	{
		unsigned int kmax = hdedw[l].size();
		for(unsigned int k = 0; k < kmax; k++)
		{
			unsigned int imax = du[l].size();
			unsigned int i = k % imax;
			unsigned int j = k / imax;
			hdedw[l][k] = hdelta[l][i] * hz[l - 1][j];
		}
		float max_error = compareVector(hdedw[l], ddedw[l]);
		std::cout << "max_error(dedw[" << l << "]) = " << max_error << std::endl;
	}
}
/////////////////////////////////////////////////////////////////////////////////
class BackpropagationNumericDiffTest :
	public ::testing::Test ,
	public ::testing::WithParamInterface<std::tuple<unsigned int, unsigned int, unsigned int, float>>
{
protected:
	void SetUp(){}
	void TearDown(){}
};

//std::vector<unsigned int> dimlist{1, 2, 10, 100, 1000};
//std::vector<unsigned int> dimlist{1, 2, 10, 100};
INSTANTIATE_TEST_CASE_P
	(
		InstantiateBackpropagationNumericDiffTest,
		BackpropagationNumericDiffTest,
		::testing::Combine
			(
				::testing::ValuesIn(std::vector<unsigned int>{1u, 2u, 10u, 100u, 1025u}),
				::testing::ValuesIn(std::vector<unsigned int>{1u, 2u, 10u}),
				::testing::ValuesIn(std::vector<unsigned int>{1u, 2u, 10u, 100u}),
				::testing::ValuesIn(std::vector<float>{0.0625f, 0.03125f})
				//::testing::ValuesIn(std::vector<float>{0.015625f, std::sqrt(std::numeric_limits<float>::epsilon())})
				//::testing::ValuesIn(std::vector<float>{0.0625f, 0.03125f, 0.015625f})
			)
	);
float ErrorFunc(const std::vector<float>& y, const std::vector<float>& d)
{
	return 0.5f * std::inner_product
		(
			y.begin(), y.end(),
			d.begin(),
			0.0f,
			[](float x_, float y_){return x_ + y_;},
			[](float x_, float y_){return (x_ - y_) * (x_ - y_);}
		);
}

//パラメータの更新に使用するdEdW, dEdbが正しいかを数値微分と比較して確認する
TEST_P(BackpropagationNumericDiffTest, NumericDifferentiation)
{
	//乱数の初期化
	std::random_device rdev;
	std::mt19937 engine(rdev());
	std::uniform_real_distribution<float> urd(0.0f, 1.0f);
	
	//BPの初期化
	unsigned int d0 = std::get<0>(GetParam());
	unsigned int d1 = std::get<1>(GetParam());
	std::vector<unsigned int> unit_count{d0, d1, d0};
	Backpropagation b(unit_count.size());
	b.init(unit_count);
	
	//数値微分との差の評価に使用するepsilon
	float epsilon0 = 1.0f;
	//数値微分に使用するepsilon
	float epsilon = std::get<3>(GetParam());
	
	
	std::vector<float> x(unit_count[0], urd(engine));
	std::vector<float> y0;
	unsigned int sample_count = std::get<2>(GetParam());
	for(unsigned int n = 0; n < sample_count; n++)
	{
		x = std::vector<float>(unit_count[0], urd(engine));
		//順伝播
		b.forward(x, y0);
		//逆伝播
		b.back(x);
		
		if(n != sample_count - 1)
		{
			//パラメータの更新
			b.updateParameter();
		}
	}
	
	//この時点でdEdW, dEdbが算出された
	
	//E0 = 0.5f *  ||y0 -x||^2
	float E0 = ErrorFunc(y0, x);
	
	//微分の基準となるパラメータの取得
	auto w0 = b.getWeightAsVector();
	auto w  = w0;
	auto bias0 = b.getBiasAsVector();
	auto bias  = bias0;
	
	//比較用のdEdWの取得
	auto dedw = b.getDEDWAsVector();
	//比較用のdEdbの取得
	auto dedb = b.getDEDBAsVector();
	
	unsigned int imax = w.size();
	for(unsigned int i = 0; i < imax; i++)
	{
		unsigned int jmax = w[i].size();
		for(unsigned int j = 0; j < jmax; j++)
		{
			//dEdWの数値微分を算出する
			{
				w = w0;
				float e = std::abs(w[i][j]);
				e = (e < 1.0f) ? 1.0f : e;
				e *= epsilon;
				w[i][j] += e;
				b.setWeight(w);
				
				std::vector<float> y;
				b.forward(x, y);
				
				float E = ErrorFunc(y, x);
				
				//下記を算出する
				//(E - E0) / epsilon
				float ndedw = (E - E0) / e;
				
				float e0 = std::abs(w[i][j]);
				e0 = (e0 < 1.0f) ? 1.0f : e0;
				e0 *= epsilon0;
				//数値微分と比較する
				EXPECT_NEAR(ndedw, dedw[i][j], e0);
				if(!(std::abs(ndedw - dedw[i][j]) < e0))
				{
					//失敗した時に情報表示する
					std::cout << "ERROR." << std::endl;
					std::cout << "epsilon  = " << epsilon  << std::endl;
					std::cout << "e        = " << e        << std::endl;
					std::cout << "epsilon0 = " << epsilon0 << std::endl;
					std::cout << "e0       = " << e0       << std::endl;
					std::cout << "d0 = " << d0 << std::endl;
					std::cout << "d1 = " << d1 << std::endl;
					std::cout << "i  = " << i  << std::endl;
					std::cout << "j  = " << j  << std::endl;
					std::cout << "E  = " << std::setprecision(10) << E  << std::endl;
					std::cout << "E0 = " << std::setprecision(10) << E0 << std::endl;
					std::cout << "ndedw  = " << ndedw << std::endl;
					std::cout << "dedw["<< i << "][" << j << "]  = " << dedw[i][j] << std::endl;
					std::cout << "diff  = " << (ndedw - dedw[i][j]) << std::endl;
					BackpropagationTest_All_showInfo(d0, sample_count, b, x, y);
					return;
				}
			}
		}
		//biasの数値微分に影響しないように
		//weightを元に戻す
		b.setWeight(w0);
		jmax = bias[i].size();
		for(unsigned int j = 0; j < jmax; j++)
		{
			//dEdbの数値微分を算出する
			bias = bias0;
			float e = std::abs(bias[i][j]);
			e = (e < 1.0f) ? 1.0f : e;
			e *= epsilon;
			bias[i][j] += e;
			b.setBias(bias);
			
			std::vector<float> y;
			b.forward(x, y);
			
			float E = ErrorFunc(y, x);
			
			//下記を算出する
			//(E - E0) / epsilon
			float ndedb = (E - E0) / e;
			float e0 = std::abs(bias[i][j]);
			e0 = (e0 < 1.0f) ? 1.0f : e0;
			e0 *= epsilon0;
			//数値微分と比較する
			EXPECT_NEAR(ndedb, dedb[i][j], e0);
			if(!(std::abs(ndedb - dedb[i][j]) < e0))
			{
				//失敗した時に情報表示する
				std::cout << "ERROR." << std::endl;
				std::cout << "epsilon  = " << epsilon  << std::endl;
				std::cout << "e        = " << e        << std::endl;
				std::cout << "epsilon0 = " << epsilon0 << std::endl;
				std::cout << "e0       = " << e0       << std::endl;
				std::cout << "d0 = " << d0 << std::endl;
				std::cout << "d1 = " << d1 << std::endl;
				std::cout << "i  = " << i  << std::endl;
				std::cout << "j  = " << j  << std::endl;
				std::cout << "E  = " << std::setprecision(10) << E  << std::endl;
				std::cout << "E0 = " << std::setprecision(10) << E0 << std::endl;
				std::cout << "ndedb  = " << ndedb << std::endl;
				std::cout << "dedb[" << i << "][" << j << "]  = " << dedb[i][j] << std::endl;
				std::cout << "diff  = " << (ndedb - dedb[i][j]) << std::endl;
				BackpropagationTest_All_showInfo(d0, sample_count, b, x, y);
				return;
			}
		}
		//weightの数値微分に影響しないように
		//biasを元に戻す
		b.setBias(bias0);
	}
}

/////////////////////////////////////////////////////////////////////////////////
class BackpropagationStreamTest :
	public ::testing::Test ,
	public ::testing::WithParamInterface<std::tuple<unsigned int, unsigned int>>
{
protected:
	void SetUp(){}
	void TearDown(){}
};

INSTANTIATE_TEST_CASE_P
	(
		InstantiateBackpropagationStreamTest,
		BackpropagationStreamTest,
		::testing::Combine
			(
				::testing::ValuesIn(std::vector<unsigned int>{2050u}),
				::testing::ValuesIn(std::vector<unsigned int>{2000u})
			)
	);
TEST(BackpropagationStreamTest, Init)
{
	Backpropagation b(3);
	b.init({3,2,3});
	
	b.setSubStreamCount(1);
	EXPECT_EQ(b.getSubStreamCount(), 1);
	b.getMainStream();
	b.getSubStream(0);
	
	b.setSubStreamCount(3);
	EXPECT_EQ(b.getSubStreamCount(), 3);
	b.getMainStream();
	b.getSubStream(0);
	b.getSubStream(1);
	b.getSubStream(2);
}


TEST_P(BackpropagationStreamTest, Evaluate)
{
	//乱数の初期化
	std::random_device rdev;
	std::mt19937 engine(rdev());
	std::uniform_real_distribution<float> urd(0.0f, 1.0f);
	
	Backpropagation b(3);
	unsigned int d0 = std::get<0>(GetParam());
	unsigned int d1 = std::get<1>(GetParam());
	b.init({d0,d1,d0});
	std::vector<float> x(d0);
	std::vector<float> y(d0);
	auto d = x;
	
	//100サイクル回す
	for(unsigned int n = 0; n < 100; n ++)
	{
		//xを乱数で初期化する
		std::transform(x.begin(), x.end(), x.begin(), [&](float){return urd(engine);});
		b.forward(x, y);
		b.back(d);
		b.updateParameter();
	}
}

/////////////////////////////////////////////////////////////////////////////////
class BackpropagationObtainDEDWTest :
	public ::testing::Test ,
	public ::testing::WithParamInterface<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>>
{
protected:
	void SetUp(){}
	void TearDown(){}
};

INSTANTIATE_TEST_CASE_P
	(
		InstantiateBackpropagationObtainDEDWTest,
		BackpropagationObtainDEDWTest,
		::testing::Combine
			(
				::testing::ValuesIn(std::vector<unsigned int>{1025}),//1, 10, 100, 1024, 1025}),//512
				::testing::ValuesIn(std::vector<unsigned int>{128}),//32, 128, 256, 512, 1024}),//256
				::testing::ValuesIn(std::vector<unsigned int>{32}),//1, 32, 128, 256, 512, 1024}),//256
				::testing::ValuesIn(std::vector<unsigned int>{32})//1, 4, 13, 31, 32})
			)
	);
TEST_P(BackpropagationObtainDEDWTest, test)
{
	std::cout << "テスト開始" << std::endl;
	
	//乱数の初期化
	//std::random_device rdev;
	//std::mt19937 engine(rdev());
	//std::uniform_real_distribution<float> urd(0.0f, 1.0f);
	
	CuBlasManager::getHandle();
	CuRandManager::getGenerator();
	
	
	unsigned int dimension    = std::get<0>(GetParam());
	unsigned int thread_count = std::get<1>(GetParam());
	unsigned int data_count   = std::get<2>(GetParam());
	unsigned int substream_count = std::get<3>(GetParam());
	
	std::cout << "乱数の初期化完了" << std::endl;
	
	std::vector<unsigned int> unit_count(64, dimension);
	Backpropagation b(unit_count.size());
	b.setSubStreamCount(substream_count);
	EXPECT_EQ(b.getSubStreamCount(), substream_count);
	b.init(unit_count);
	
	std::cout << "Backpropagation の初期化完了" << std::endl;
	
	//xを乱数で初期化する
	//std::transform(x.begin(), x.end(), x.begin(), [&](float){return urd(engine);});
	DeviceVector dx(dimension);
	CURAND_CALL(curandGenerateUniform(CuRandManager::getGenerator(), dx.getAddress(), dx.getDimension()));
	std::vector<float> x = dx.get();
	std::vector<float> y(dimension);
	std::vector<float>& d =x;
	
	std::cout << "x の初期化完了" << std::endl;
	
	b.forward(x, y);
	b.back(d);
	
	std::cout << "forward(), back()完了" << std::endl;
	
	//時刻測定イベントの定義
	cudaEvent_t start;
	cudaEvent_t stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));
	//時刻測定前に同期
	CUDA_CALL(cudaDeviceSynchronize());
	//時間測定開始
	CUDA_CALL(cudaEventRecord(start, 0));
	//obtainDEDWを単独で実行
	for(unsigned int n = 0; n < 10; n++)
	{
		for(unsigned int l = unit_count.size() - 1; l >= 1; l--)
		{
			b.obtainDEDW(l, thread_count, data_count);
		}
	}
	//時刻測定前に同期
	CUDA_CALL(cudaDeviceSynchronize());
	//時間測定完了
	CUDA_CALL(cudaEventRecord(stop, 0));
	CUDA_CALL(cudaEventSynchronize(stop));
	//実行時間
	float elapsed_time_ms;
	CUDA_CALL(cudaEventElapsedTime(&elapsed_time_ms, start, stop));
	//実行時間を表示
	std::cout << "(label)"       << '\t';
	std::cout << "dimension"       << '\t';
	std::cout << "thread_count"    << '\t';
	std::cout << "data_count"      << '\t';
	std::cout << "substream_count" << '\t';
	std::cout << "elapsed_time_ms" << std::endl;
	std::cout << "EV" << '\t';
	std::cout << dimension << '\t';
	std::cout << thread_count << '\t';
	std::cout << data_count << '\t';
	std::cout << substream_count << '\t';
	std::cout << elapsed_time_ms << std::endl;
}
/////////////////////////////////////////////////////////////////////////////////
class CudaManagerTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

TEST(CudaManagerTest, Stream)
{
	CudaManager& cm = CudaManager::getInstance();
	cm.initStream(2);
	cm.getStream(0);
	cm.getStream(1);
	EXPECT_EQ(cm.getStreamCount(), 2);
	
	cm.initStream(5);
	cm.getStream(0);
	cm.getStream(1);
	cm.getStream(2);
	cm.getStream(3);
	cm.getStream(4);
	EXPECT_EQ(cm.getStreamCount(), 5);
}

/////////////////////////////////////////////////////////////////////////////////
class CuRandManagerTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

TEST(CuRandManagerTest, Constructor)
{
	CuRandManager::getGenerator();
}

TEST(CuRandManagerTest, Generate)
{
	DeviceVector dv0(1);
	CURAND_CALL(curandGenerateUniform(CuRandManager::getGenerator(), dv0.getAddress(), dv0.getDimension()));
	DeviceVector dv1(100);
	CURAND_CALL(curandGenerateUniform(CuRandManager::getGenerator(), dv1.getAddress(), dv1.getDimension()));
}
/////////////////////////////////////////////////////////////////////////////////
class CuBlasFunctionTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

/////////////////////////////////////////////////////////////////////////////////
class CuBlasFunctionTest_1V :
	public ::testing::Test,
	public ::testing::WithParamInterface<unsigned int>
{
protected:
	void SetUp(){}
	void TearDown(){}
};

INSTANTIATE_TEST_CASE_P
	(
		InstantiateCuBlasFunctionTest_1V,
		CuBlasFunctionTest_1V,
		::testing::ValuesIn(std::vector<unsigned int>{1,10,100})
	);

///////////////////////////////////////
class CuBlasFunctionTest_2V :
	public ::testing::Test,
	public ::testing::WithParamInterface<std::tuple<unsigned int, unsigned int> >
{
protected:
	void SetUp(){}
	void TearDown(){}
};

INSTANTIATE_TEST_CASE_P
	(
		InstantiateCuBlasFunctionTest_2V,
		CuBlasFunctionTest_2V,
		::testing::Combine
			(
				::testing::ValuesIn(std::vector<unsigned int>{1, 10, 100, 1025}),
				::testing::ValuesIn(std::vector<unsigned int>{1, 10, 100, 1025})
			)
	);
///////////////////////////////////////

TEST_P(CuBlasFunctionTest_1V, Sscal_Vector)
{
	unsigned int dimension = GetParam();
	std::vector<float> x(dimension, 0.0f);
	//xを乱数で初期化する
	
	DeviceVector x_d(x);
	CURAND_CALL
		(
			curandGenerateUniform
				(
					CuRandManager::getGenerator(),
					x_d.getAddress(),
					x_d.getDimension()
				)
		);
	x = x_d.get();
	float alpha = 0.5f;
	Sscal(&alpha, x_d);
	
	
	std::transform(x.begin(), x.end(), x.begin(), [&](float _x){return _x *= alpha;});
	
	auto y = x_d.get();
	
	float diff = compareVector(x, y);
	std::cout << "diff = " << diff << std::endl;
}

TEST_P(CuBlasFunctionTest_2V, Sscal_Matrix)
{
	unsigned int N = std::get<0>(GetParam());
	unsigned int M = std::get<1>(GetParam());
	
	std::vector<float> A(M * N, 0.0f);
	//Aを乱数で初期化する
	DeviceMatrix A_d(M, N, A);
	CURAND_CALL
		(
			curandGenerateUniform
				(
					CuRandManager::getGenerator(),
					A_d.getAddress(),
					A_d.getRowCount() * A_d.getColumnCount()
				)
		);
	A = A_d.get();
	
	float alpha = 0.1f;
	Sscal(&alpha, A_d);
	
	
	std::transform(A.begin(), A.end(), A.begin(), [&](float _x){return _x *= alpha;});
	
	auto B = A_d.get();
	
	
	
	//std::cout << "A = (";
	//std::for_each(A.begin(), A.end(), [](float _x){std::cout << _x << ", ";});
	//std::cout << ")" << std::endl;
	//std::cout << "B = (";
	//std::for_each(B.begin(), B.end(), [](float _x){std::cout << _x << ", ";});
	//std::cout << ")" << std::endl;
	
	float diff = compareVector(A, B);
	std::cout << "diff = " << diff << std::endl;
}

TEST(CuBlasFunctionTest, Sdgmm)
{
	DeviceMatrix A(3, 4, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
	DeviceVector x{1.0f, 2.0f, 3.0f, 4.0f};
	DeviceVector y{1.0f, 2.0f, 3.0f};
	DeviceMatrix C(3, 4);
	DeviceMatrix D(3, 4);
	std::vector<float> Ax
		{
			 1.0f,  2.0f,  3.0f,
			 8.0f, 10.0f, 12.0f,
			21.0f, 24.0f, 27.0f,
			40.0f, 44.0f, 48.0f
		};
	std::vector<float> yA
		{
			 1.0f,  4.0f,  9.0f,
			 4.0f, 10.0f, 18.0f,
			 7.0f, 16.0f, 27.0f,
			10.0f, 22.0f, 36.0f
		};
	
	Sdgmm(A, x, C);
	Sdgmm(y, A, D);
	
	auto dC = C.get();
	auto dD = D.get();
	
	compareVector(dC, Ax);
	compareVector(dD, yA);
}
/////////////////////////////////////////////////////////////////////////////////
class CuSolverDnTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

TEST(CuSolverDnTest, getHandle)
{
	CuSolverDnManager::getHandle();
}

TEST(CuSolverDnTest, DnSsyevd)
{
	DeviceMatrix dA(3, 3, {3.5f, 0.5f, 0.0f, 0.5f, 3.5f, 0.0f, 0.0f, 0.0f, 2.0f});
	DeviceVector dW;
	DeviceMatrix dV;
	DnSsyevd(dA, dW, dV);
	auto hW = dW.get();
	auto hV = dV.get();
	
	float rt2d2 = std::sqrt(2.0f) / 2.0f;
	std::vector<float> W{2.0f, 3.0f, 4.0f};
	std::vector<float> V{0.0f, 0.0f, 1.0f, - rt2d2, rt2d2, 0.0f, rt2d2, rt2d2, 0.0f};
	
	compareVector(hW, W);
	compareVector(hV, V);
}

/////////////////////////////////////////////////////////////////////////////////
class NormalizationTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

void printVector(const std::vector<float>& v, const std::string& vname)
{
	std::cout << vname << " = {";
	for(auto&& x : v)
	{
		std::cout << std::showpoint << std::setprecision(8)<< x << ", ";
	}
	std::cout << "}" << std::endl;
}

TEST(NormalizationTest, simple)
{
	//正規化用のクラス
	Normalization n;
	//元のデータ
	DeviceMatrix dX(2,3,{3.0f, 6.0f, 6.0f, 12.0f, 9.0f, 3.0f});
	//算出される値
	std::vector<float> mean{6.0f, 7.0f};
	std::vector<float> varcovmatrix{6.0f, -3.0f, -3.0f, 14.0f};
	std::vector<float> varcovvalue{5.0f, 15.0f};
	std::vector<float> P_pca{-0.424264068711929f, -0.0816496580927726f, -0.14142135623731f, 0.244948974278318f};
	std::vector<float> Y_pca
		{
			 1.4142135623731f,    0.0f, 
			-0.707106781186547f,  1.22474487139159f, 
			-0.707106781186547f, -1.22474487139159f
		};

	std::vector<float> P_zca{0.428312124924678f, 0.0567044117258391f, 0.0567044117258391f, 0.277100360322441f};
	std::vector<float> Y_zca
		{
			-1.34164078649987f, -0.447213595499958f,
			 0.283522058629195f, 1.3855018016122f,
			 1.05811872787068f, -0.938288206112246f
		};

	
	//元データで初期化
	n.init(dX);
	
	//平均を算出して比較
	auto dmean = n.getMean();
	EXPECT_EQ(compareVector(mean, dmean.get()) < 0.0625f, true);
	
	//分散共分散行列を算出して比較
	auto dvarcovmatrix = n.getVarCovMatrix().get();
	//下半分の値が不定値なので上半分の値を設定する
	dvarcovmatrix[1] = dvarcovmatrix[2];
	EXPECT_EQ(compareVector(varcovmatrix, dvarcovmatrix) < 0.0625f, true);
	printVector( varcovmatrix, " varcovmatrix");
	printVector(dvarcovmatrix, "dvarcovmatrix");
	
	//分散共分散行列の固有値を比較
	auto dvarcovvalue = n.getVarCovEigenValue().get();
	EXPECT_EQ(compareVector(varcovvalue, dvarcovvalue) < 0.0625f, true);
	printVector( varcovvalue, " varcovvalue");
	printVector(dvarcovvalue, "dvarcovvalue");
	
	//PCA白色化
	//白色化変換行列を取得
	auto dP_pca = n.getPCAWhiteningMatrix().get();
	
	//白色化を実行
	auto dY_pca =  n.getPCAWhitening(dX).get();
	//結果を比較
	EXPECT_EQ(compareVector(P_pca, dP_pca) < 0.0625f, true);
	printVector(  P_pca, " P_pca");
	printVector( dP_pca, "dP_pca");
	EXPECT_EQ(compareVector(Y_pca, dY_pca) < 0.0625f, true);
	printVector(  Y_pca, " Y_pca");
	printVector( dY_pca, "dY_pca");
	
	
	
	//ZCA白色化
	//白色化変換行列を取得
	auto dP_zca = n.getZCAWhiteningMatrix().get();
	//白色化を実行
	auto dY_zca = n.getZCAWhitening(dX).get();
	//結果を比較
	EXPECT_EQ(compareVector(P_zca, dP_zca) < 0.0625f, true);
	printVector(  P_zca, " P_zca");
	printVector( dP_zca, "dP_zca");
	EXPECT_EQ(compareVector(Y_zca, dY_zca) < 0.0625f, true);
	printVector(  Y_zca, " Y_zca");
	printVector( dY_zca, "dY_zca");
}

///////////////////////////////////////
class NormalizationGeneralTest :
	public ::testing::Test,
	public ::testing::WithParamInterface<std::tuple<unsigned int, unsigned int> >
{
protected:
	void SetUp(){}
	void TearDown(){}
};

INSTANTIATE_TEST_CASE_P
	(
		InstantiateNormalizationGeneralTest,
		NormalizationGeneralTest,
		::testing::Combine
			(
				::testing::ValuesIn(std::vector<unsigned int>{1, 10, 100, 1000}),
				::testing::ValuesIn(std::vector<unsigned int>{10000})
			)
	);

TEST_P(NormalizationGeneralTest, test)
{
	unsigned int D = std::get<0>(GetParam());
	unsigned int N = std::get<1>(GetParam());
	//std::cout << "(D, N) = (" << D << ", " << N << ")" << std::endl;
	DeviceMatrix X(D, N);
	CURAND_CALL(curandGenerateUniform(CuRandManager::getGenerator(), X.getAddress(), D * N));
	if(D == 2)
	{
		std::vector<float> v;
		for(unsigned int i = 0; i < N; i++)
		{
			float x = static_cast<float>(i) / static_cast<float>(N);
			float y = std::sin(4.0f * x);
			v.push_back(x);
			v.push_back(y);
		}
		X = DeviceMatrix(D, N, v);
	}
	
	std::vector<float> unit_matrixD;
	for(unsigned int j = 0; j < D; j++)
	{
		for(unsigned int i = 0; i < D; i++)
		{
			float x = (i != j) ? 0.0f : 1.0f;
			unit_matrixD.push_back(x);
		}
	}
	
	//printVector(X.get(), "X");
	Normalization n;
	
	n.init(X);
	//printVector(n.getMean().get(),               "Mean              ");
	//printVector(n.getVarCovMatrix().get(),       "VarCovMatrix      ");
	//printVector(n.getPCAWhiteningMatrix().get(), "PCAWhiteningMatrix");
	
	DeviceMatrix Y_pca = n.getPCAWhitening(X);
	//printVector(Y_pca.get(),                     "Y_pca"             );
	
	float max_error = 0.0625;
	
	Normalization n_pca;
	n_pca.init(Y_pca);
	std::cout << "T_pca" << std::endl;
	//下半分に上半分と同一の値を設定する
	auto VarCovMatrix_pca = n_pca.getVarCovMatrix().get();
	for(unsigned int j = 0; j < D; j++)
	{
		for(unsigned int i = 0; i < D; i++)
		{
			if(j < i)
			{
				VarCovMatrix_pca[i + j * D] = VarCovMatrix_pca[j + i * D];
			}
		}
	}
	
	float diff = compareVector(VarCovMatrix_pca, unit_matrixD, max_error);
	
	if(diff >= max_error)
	{
		printVector(n.getMean().get(),                   "Mean                  ");
		printVector(n.getVarCovMatrix().get(),           "VarCovMatrix          ");
		printVector(n.getPCAWhiteningMatrix().get(),     "PCAWhiteningMatrix    ");
		//printVector(Y_pca.get(),                       "Y_pca                 ");
		printVector(n_pca.getMean().get(),               "Mean_pca              ");
		printVector(VarCovMatrix_pca,                    "VarCovMatrix_pca      ");
		printVector(n_pca.getPCAWhiteningMatrix().get(), "PCAWhiteningMatrix_pca");
	}
	
	DeviceMatrix Y_zca = n.getZCAWhitening(X);
	//printVector(Y_zca.get(),                     "Y_zca"             );
	
	Normalization n_zca;
	n_zca.init(Y_zca);
	std::cout << "T_zca" << std::endl;
	//下半分に上半分と同一の値を設定する
	auto VarCovMatrix_zca = n_zca.getVarCovMatrix().get();
	for(unsigned int j = 0; j < D; j++)
	{
		for(unsigned int i = 0; i < D; i++)
		{
			if(j < i)
			{
				VarCovMatrix_zca[i + j * D] = VarCovMatrix_zca[j + i * D];
			}
		}
	}
	diff = compareVector(VarCovMatrix_zca, unit_matrixD);
	
	if(diff >= max_error)
	{
		printVector(n.getMean().get(),                   "Mean                  ");
		printVector(n.getVarCovMatrix().get(),           "VarCovMatrix          ");
		printVector(n.getPCAWhiteningMatrix().get(),     "PCAWhiteningMatrix    ");
		//printVector(Y_zca.get(),                       "Y_zca                 ");
		printVector(n_zca.getMean().get(),               "Mean_zca              ");
		printVector(VarCovMatrix_zca,                    "VarCovMatrix_zca      ");
		printVector(n_zca.getZCAWhiteningMatrix().get(), "ZCAWhiteningMatrix_zca");
	}
	
	if(false && (D == 2))
	{
		auto vX = X.get();
		auto vY_pca = Y_pca.get();
		auto vY_zca = Y_zca.get();
		
		for(unsigned int i = 0; i < N; i++)
		{
			unsigned int ix = i * 2;
			unsigned int iy = i * 2 + 1;
			std::cout << vX[ix]     << ", " << vX[iy]     << ", ";
			std::cout << vY_pca[ix] << ", " << vY_pca[iy] << ", ";
			std::cout << vY_zca[ix] << ", " << vY_zca[iy];
			std::cout <<std::endl;
		}
	}

}

/////////////////////////////////////////////////////////////////////////////////

TEST(NormalizationTest, csv)
{
	//CSVファイルから学習データ(行列)の読み込み
	//CSVファイル
	std::string csv_file_name("../../../anaconda3/test_anaconda/data.csv");
	
	//学習データを格納するvector
	std::vector<float> data;
	//学習データの次元(CSVファイルのカラム数=行列の"行"数(縦横入れ替わる))
	int dimension = 0;
	
	//CSVファイルを読み込み用に開く
	std::ifstream csv_file(csv_file_name);
	//1行読み飛ばす
	std::string dummy;
	csv_file >> dummy;
	//ファイルを1行ずつ読み込み
	while(!csv_file.eof())
	{
		//1行ごとの処理
		//1行分の文字列の取得
		std::string line;
		csv_file >> line;
		
		std::cout << line << std::endl;
		
		//1行分の文字列ストリーム
		std::istringstream line_stream(line);
		//トークン（カンマ区切り）
		std::string token;
		//カラム数のカウンタ
		int column_count = 0;
		//カンマで区切られた文字列を格納
		while(std::getline(line_stream, token, ','))
		{
			//トークンをfloatに変換
			float token_float = std::stof(token);
			//vectorに格納
			data.push_back(token_float);
			//カラム数をカウントする
			column_count++;
		}
		//次元が未設定の時
		if(dimension == 0)
		{
			//次元としてカラム数を設定する
			dimension = column_count;
		}
	}
	//CSVファイルのデータの行数 = 行列の列数
	int data_length = data.size() / dimension;
	
	//学習データをDeviceMatrixとして初期化
	DeviceMatrix dX(dimension, data_length, data);

	//Normalizationを初期化して学習データで初期化
	Normalization n;
	//白色化の変換行列の生成
	n.init(dX);
	
	//初期値自体の白色化の実行
	DeviceMatrix nX = n.getPCAWhitening(dX);
	
	//白色化の結果を結果のCSV出力
	std::vector<float> data_w = nX.get();
	
	std::string result_csv_file_name("../../../anaconda3/test_anaconda/data_w_cpp.csv");
	std::ofstream result_csv_file(result_csv_file_name.c_str());
	for(int i = 0; i < data_length; i++)
	{
		for(int j = 0; j < dimension; j++)
		{
			result_csv_file << data_w[i * dimension + j];
			if(j != (dimension - 1))
			{
				result_csv_file << ',';
			}
		}
		result_csv_file << std::endl;
	}
	
	printVector(dX.get(),                            "dX                ");
	printVector(nX.get(),                            "nX                ");
	printVector(n.getMean().get(),                   "Mean              ");
	printVector(n.getVarCovMatrix().get(),           "VarCovMatrix      ");
	printVector(n.getVarCovEigenValue().get(),       "VarCovEigenValue  ");
	printVector(n.getVarCovEigenVector().get(),      "VarCovEigenVector ");
	printVector(n.getPCAWhiteningMatrix().get(),     "PCAWhiteningMatrix");
	
}

/////////////////////////////////////////////////////////////////////////////////
class BackpropagationMiniBatchTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

//ミニバッチに対応したBackpropagationのテスト
TEST(BackpropagationMiniBatchTest, Simple)
{
	Backpropagation b(3);
	unsigned int dimension = 3;
	unsigned int layer_size = 2;
	unsigned int minibatch_size = 100;
	//値を入れ替えるインデックス1
	unsigned int index_swap1 = 21;
	//値を入れ替えるインデックス2
	unsigned int index_swap2 = 65;
	
	
	std::cout << "dimension = " << dimension << std::endl;
	std::cout << "layer_size = " << layer_size << std::endl;
	std::cout << "minibatch_size = " << minibatch_size << std::endl;
	
	
	//init()ではミニバッチのサイズを引数にとる
	b.init({dimension, layer_size, dimension}, minibatch_size);
	
	//インプットをミニバッチとして定義
	DeviceMatrix X(dimension, minibatch_size);
	//一様分布に従う確率変数の値を格納
	setRandomUniform(-1.0f, 1.0f, X);
	
	DeviceMatrix Y;
	DeviceMatrix& D = X;
	
	//順伝播・逆伝播はDeviceMatrixを引数にとる
	//順伝播の実行
	b.forward(X, Y);
	
	//ミニバッチの列を入れ替えても列同士は同じ値を返すかをテスト
	auto x  = X.get();
	auto x1 = x;
	auto y  = Y.get();
	auto y1 = y;
	for(unsigned int j = 0; j < dimension; j++)
	{
		unsigned int s1 = index_swap1 * dimension + j;
		unsigned int s2 = index_swap2 * dimension + j;
		x1[s1] = x[s2];
		x1[s2] = x[s1];
		y1[s1] = y[s2];
		y1[s2] = y[s1];
	}
	
	DeviceMatrix X1(dimension, minibatch_size, x1);
	DeviceMatrix Y1;
	b.forward(X1, Y1);
	auto y1_out = Y1.get();
	
	compareVector(y1, y1_out, 0.0f);
	printVector(y1, "y1");
	printVector(y1_out, "y1_out");
	
	//DeviceMatrixでback(), updateParameter() が実行できるかをテスト
	//逆伝播の実行
	b.back(D);
	
	auto delta1 = b.getDelta()[1].get();
	
	DeviceMatrix& Z0 = X;
	auto z0 = Z0.get();
	
	//TODO weightについて意図した更新のデータを作成する
	auto weight1_old = b.getWeight()[1].get();
	auto weight1 = b.getWeight()[1].get();
	//delta1 * (Z0 ^ T)
	for(unsigned int j = 0; j < dimension; j++)
	{
		for(unsigned int i = 0; i < layer_size; i++)
		{
			float delta_weight1_ij = 0.0f;
			for(unsigned int k = 0; k < minibatch_size; k++)
			{
				delta_weight1_ij += delta1[k * layer_size + i] * z0[k * dimension + j];
			}
			delta_weight1_ij /= static_cast<float>(minibatch_size);
			delta_weight1_ij *= b.getEpsilon();
			weight1[j * layer_size + i] -= delta_weight1_ij;
		}
	}
	
	
	
	std::vector<float> bias1_old = b.getBias()[1].get();
	std::vector<float> bias1     = b.getBias()[1].get();
	for(unsigned int i = 0; i < layer_size; i++)
	{
		float delta_bias1_i = 0.0f;
		for(unsigned int j = 0; j < minibatch_size; j++)
		{
			delta_bias1_i += delta1[j * layer_size + i];
		}
		delta_bias1_i /= static_cast<float>(minibatch_size);
		delta_bias1_i *= b.getEpsilon();
		bias1[i] -= delta_bias1_i;
	}
	
	//パラメータの更新
	b.updateParameter();
	
	printVector(delta1,    "delta1");
	
	//TODO weightについて結果を比較して差分が十分小さいことを確認する
	auto weight1_out = b.getWeight()[1].get();
	compareVector(weight1, weight1_out);
	printVector(weight1_old, "weight1_old");
	printVector(weight1,     "weight1    ");
	printVector(weight1_out, "weight1_out");
	
	
	auto bias1_out = b.getBias()[1].get();
	compareVector(bias1, bias1_out);
	printVector(bias1_old, "bias1_old");
	printVector(bias1,     "bias1    ");
	printVector(bias1_out, "bias1_out");
}

/////////////////////////////////////////////////////////////////////////////////
class AutoEncoderTest :
	public ::testing::Test
{
protected:
	void SetUp(){}
	void TearDown(){}
};

//ミニバッチに対応したBackpropagationのテスト
TEST(AutoEncoderTest, Simple)
{
	
	//正規化用データ
	std::vector<float> normarize_input_base(20);
	for(int i = 0; i < 10; i++)
	{
		float x = static_cast<float>(i);
		normarize_input_base[2 * i    ] = x;
		normarize_input_base[2 * i + 1] = x * x;
	}
	
	//正規化用データのDeviceMatrix
	DeviceMatrix normarize_input(2, 10);
	normarize_input.set(normarize_input_base);
	
	AutoEncoder a;
	a.init(normarize_input, 5, 10);
	
	//学習用データ
	std::vector<float> minibatch_input_base(20);
	for(int i = 0; i < 10; i++)
	{
		float x = static_cast<float>(i) + 0.5f;
		minibatch_input_base[2 * i    ] = x;
		minibatch_input_base[2 * i + 1] = x * x;
	}
	
	//学習用データのDeviceMatrix
	DeviceMatrix minibatch_input(2, 10);
	minibatch_input.set(minibatch_input_base);
	
	//学習の実行
	for(int i = 0; i < 10000; i++)
	{
		a.learning(minibatch_input);
	}
	//出力の取得
	auto minibatch_output = a.learning(minibatch_input);
	
	//normarize_inputの表示
	std::cout << "normarize_input" << std::endl;
	for(int i = 0; i < 10; i++)
	{
		std::cout << normarize_input_base[2 * i    ] << "\t";
		std::cout << normarize_input_base[2 * i + 1] << std::endl;
	}
	//minibatch_inputの表示
	auto minibatch_input_vector = minibatch_input.get();
	std::cout << "minibatch_input" << std::endl;
	for(int i = 0; i < 10; i++)
	{
		std::cout << minibatch_input_vector[2 * i    ] << "\t";
		std::cout << minibatch_input_vector[2 * i + 1] << std::endl;
	}
	//minibatch_outputの表示
	auto minibatch_output_vector = minibatch_output.get();
	std::cout << "minibatch_output" << std::endl;
	for(int i = 0; i < 10; i++)
	{
		std::cout << minibatch_output_vector[2 * i    ] << "\t";
		std::cout << minibatch_output_vector[2 * i + 1] << std::endl;
	}
	EXPECT_TRUE(false);//今の処理では学習していないので注意喚起のために失敗させる
}



//////////////////////////////////////////////////////////////////////
// main()
//////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	//::testing::GTEST_FLAG(filter)="-:*NumericDifferentiation*";
	
	//::testing::GTEST_FLAG(filter)="*BackpropagationObtainDEDWTest*";
	
	//::testing::GTEST_FLAG(filter)="*AutoEncoderTest*";
	::testing::GTEST_FLAG(filter)="*NormalizationTest.csv*";
	//::testing::GTEST_FLAG(filter)="*NormalizationGeneralTest*";
	//::testing::GTEST_FLAG(filter)="*Sdgmm*";
	//::testing::GTEST_FLAG(filter)="*CuSolverDnTest*";
	//::testing::GTEST_FLAG(filter)="*CuRandManagerTest*";
	//::testing::GTEST_FLAG(filter)="*CuBlasFunctionTest_2V*";
	//::testing::GTEST_FLAG(filter)="*Evaluate*";
	//::testing::GTEST_FLAG(filter)="*All*:*Simple*";
	//::testing::GTEST_FLAG(filter)="*Input*:*Output*";
	//::testing::GTEST_FLAG(filter)="*BackpropagationMiniBatchTest*";
	//::testing::GTEST_FLAG(filter)="*Backpropagation*";
	
	
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

