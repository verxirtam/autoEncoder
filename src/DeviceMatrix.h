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


#include "CUBLASManager.h"

class DeviceMatrix
{
private:
	const int rowCount;
	const int columnCount;
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
	DeviceMatrix(int r, int c):
		rowCount(r),
		columnCount(c),
		device(nullptr)
	{
		cudaMalloc((void**)&device, rowCount * columnCount * sizeof(float));
		
	}
	~DeviceMatrix()
	{
		cudaFree(device);
	}
	void set(const float* host)
	{
		cublasStatus_t stat;
		stat = cublasSetMatrix(rowCount, columnCount, sizeof(float), host, rowCount, device, rowCount);
		if(stat != CUBLAS_STATUS_SUCCESS)
		{
			std::cout << "error at cublasSetMatrix()" << std::endl;
		}
	}
	void get(float* host)
	{
		cublasStatus_t stat;
		stat = cublasGetMatrix(rowCount, columnCount, sizeof(float), device, rowCount, host, rowCount);
		if(stat != CUBLAS_STATUS_SUCCESS)
		{
			std::cout << "error at cublasSetMatrix()" << std::endl;
		}
		
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


