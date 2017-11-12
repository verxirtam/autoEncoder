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
	//1データの長さ(過去何分のデータを使用するか)
	unsigned int timeLength;
	//正規化用の情報を取得する
	void getNormarizeInput(DeviceMatrix& normarize_input);
public:
	FXAutoEncoder():
		autoEncoder(),
		dbFileName(""),
		timeLength(10)
	{
	}
	void init(const std::string& db_file_name, unsigned int time_length, unsigned int layer_size, unsigned int minibatch_size);
	const AutoEncoderType& getAutoEncoder()
	{
		return autoEncoder;
	}
	void learning();
};

