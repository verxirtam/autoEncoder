/*
 * =====================================================================================
 *
 *       Filename:  DeviceVectorUtils.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年10月03日 00時46分03秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "DeviceVectorUtils.h"

//#include <algorithm>
#include <numeric>

namespace cuda
{

float compareVector(const std::vector<float>& x0, const std::vector<float>& x1)
{
	//要素数が異なる場合は大きい値を返す
	if(x0.size() != x1.size())
	{
		return 65536.0f;
	}
	//各成分の差の絶対値の最大値を返す
	float diff = std::inner_product
		(
		 x0.begin(), x0.end(), x1.begin(), 0.0f,
		 [](float _x, float _y){return std::max(_x, _y);},
		 [](float _x, float _y){return std::abs(_x - _y);}
		);
	
	return diff;
}

float compare(const DeviceMatrix& m0, const DeviceMatrix& m1)
{
	//行数または列数が異なる場合は大きい値を返す
	if(m0.getRowCount() != m1.getRowCount())
	{
		return 65536.0f;
	}
	if(m0.getColumnCount() != m1.getColumnCount())
	{
		return 65536.0f;
	}
	//std::vector<float>に直して比較する
	return compareVector(m0.get(), m1.get());
}

float compare(const DeviceVector& v0, const DeviceVector& v1)
{
	//std::vector<float>に直して比較する
	return compareVector(v0.get(), v1.get());
}

DeviceVector& readFromCsvFile(const std::string& csvFileName, DeviceVector& deviceVector)
{
	
	throw 1;
	
	
	return deviceVector;
}

DeviceMatrix& readFromCsvFile(const std::string& csvFileName, DeviceMatrix& deviceMatrix)
{
	
	//CSVファイルから行列の読み込み
	
	//学習データを格納するvector
	std::vector<float> data;
	//学習データの次元(CSVファイルのカラム数=行列の"行"数(縦横入れ替わる))
	int dimension = 0;
	
	//ファイルオブジェクトの作成
	std::ifstream csv_file;
	//ファイルが存在しない時は例外発生するように設定する
	csv_file.exceptions(std::ifstream::failbit);
	//ファイルオープン
	csv_file.open(csvFileName);
	
	//ファイルオープンに成功したので例外発生の設定をクリアする
	csv_file.exceptions(std::ifstream::goodbit);
	
	//ファイルを1行ずつ読み込み
	while(!csv_file.eof())
	{
		//1行ごとの処理
		//1行分の文字列の取得
		std::string line;
		csv_file >> line;
		
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
	
	//DeviceMatrixとして初期化
	deviceMatrix = DeviceMatrix(dimension, data_length, data);
	
	return deviceMatrix;
}

void writeToCsvFile(const std::string& csvFileName, const DeviceVector& deviceVector)
{
	std::vector<float> data = deviceVector.get();

	unsigned int column_count = 1;
	unsigned int row_count    = deviceVector.getDimension();
	
	std::ofstream csv_file(csvFileName);
	for(unsigned int i = 0; i < column_count; i++)
	{
		for(unsigned int j = 0; j < row_count; j++)
		{
			csv_file << data[i * row_count + j];
			if(j != (row_count - 1))
			{
				csv_file << ',';
			}
		}
		csv_file << std::endl;
	}
}

void writeToCsvFile(const std::string& csvFileName, const DeviceMatrix& deviceMatrix)
{
	std::vector<float> data = deviceMatrix.get();
	
	unsigned int column_count = deviceMatrix.getColumnCount();
	unsigned int row_count    = deviceMatrix.getRowCount();
	
	std::ofstream csv_file(csvFileName);
	for(unsigned int i = 0; i < column_count; i++)
	{
		for(unsigned int j = 0; j < row_count; j++)
		{
			csv_file << data[i * row_count + j];
			if(j != (row_count - 1))
			{
				csv_file << ',';
			}
		}
		csv_file << std::endl;
	}
}

}

