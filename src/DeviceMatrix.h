/*
 * =====================================================================================
 *
 *       Filename:  DeviceMatrix.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月19日 02時37分42秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include <vector>

#include "CUBLASManager.h"

class DeviceMatrix
{
private:
	unsigned int rowCount;
	unsigned int columnCount;
	float* device;
	//ポインタメンバを持つがコピー不要なので禁止する
	//本当はデバイスメモリ同士でコピーしたい...。
	//コピーコンストラクタ
	DeviceMatrix(const DeviceMatrix&) = delete;
	//コピー代入演算子
	DeviceMatrix& operator=(const DeviceMatrix&) = delete;
	//ムーブコンストラクタ
	DeviceMatrix(DeviceMatrix&&) = delete;
	//ムーブ代入演算子
	DeviceMatrix& operator=(DeviceMatrix&&) = delete;
public:
	//コンストラクタ
	DeviceMatrix(int r, int c):
		rowCount(r),
		columnCount(c),
		device(nullptr)
	{
		if(rowCount * columnCount != 0)
		{
			cudaMalloc((void**)&device, rowCount * columnCount * sizeof(float));
		}
	}
	//コンストラクタ(vector版)
	DeviceMatrix(int r, int c, const std::vector<float>& d):
		DeviceMatrix(r, c)
	{
		this->set(d.data());
	}
	//デフォルトコンストラクタ
	DeviceMatrix():
		rowCount(0),
		columnCount(0),
		device(nullptr)
	{
	}
	
	//デストラクタ
	~DeviceMatrix()
	{
		cudaFree(device);
	}
	void set(const float* host)
	{
		if(rowCount * columnCount == 0)
		{
			return;
		}
		cublasStatus_t stat;
		stat = cublasSetMatrix(rowCount, columnCount, sizeof(float), host, rowCount, device, rowCount);
		if(stat != CUBLAS_STATUS_SUCCESS)
		{
			std::cout << "error at cublasSetMatrix() stat = ";
			std::cout << CUBLASManager::getErrorString(stat) << std::endl;
			std::cout << "      at DeviceMatrix::set(const float*) ";
			std::cout << "rowCount = " << rowCount << ", ";
			std::cout << "columnCount = " << columnCount << std::endl;
		}
	}
	void set(const std::vector<float>& host)
	{
		this->set(host.data());
	}
	void get(float* host)
	{
		if(rowCount * columnCount == 0)
		{
			return;
		}
		cublasStatus_t stat;
		stat = cublasGetMatrix(rowCount, columnCount, sizeof(float), device, rowCount, host, rowCount);
		if(stat != CUBLAS_STATUS_SUCCESS)
		{
			std::cout << "error at cublasGetMatrix() stat = ";
			std::cout << CUBLASManager::getErrorString(stat) << std::endl;
			std::cout << "      at DeviceMatrix::get(const float*) ";
			std::cout << "rowCount = " << rowCount << ", ";
			std::cout << "columnCount = " << columnCount << std::endl;
		}
		
	}
	void get(std::vector<float>& host)
	{
		host.resize(rowCount * columnCount);
		this->get(host.data());
	}
	float* getAddress() const
	{
		return device;
	}
	int getRowCount() const
	{
		return rowCount;
	}
	int getColumnCount() const
	{
		return columnCount;
	}
};


