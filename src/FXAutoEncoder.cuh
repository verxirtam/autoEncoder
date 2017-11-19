#pragma once

#include "AutoEncoder.cuh"
#include "Func1to1Tanh.cuh"

#include <DBAccessor.h>

class FXAutoEncoder
{
private:
	//オートエンコーダの型
	using AutoEncoderType = AutoEncoder<Func1to1Tanh>;
	//オートエンコーダ
	AutoEncoderType autoEncoder;
	//DBファイルのパス
	std::string dbFileName;
	//DBアクセサ(学習用)
	DBAccessor dbAccessorLearning;
	//1データの長さ(過去何分のデータを使用するか)
	unsigned int timeLength;
	//学習用のクエリのキャッシュ
	std::vector<float> learningQueryCache;
	//正規化用の情報を取得する
	void getNormarizeInput(DeviceMatrix& normarize_input);
	//学習用のクエリから指定したレコード数分情報取得する
	bool selectRecord(unsigned int record_count, std::vector<float>& output);
public:
	FXAutoEncoder():
		autoEncoder(),
		dbFileName(""),
		dbAccessorLearning(),
		timeLength(10),
		learningQueryCache()
	{
	}
	void init(const std::string& db_file_name, unsigned int time_length, unsigned int layer_size, unsigned int minibatch_size);
	const AutoEncoderType& getAutoEncoder()
	{
		return autoEncoder;
	}
	bool learning();
};

