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
public:
	//コピーコンストラクタ
	DeviceMatrix(const DeviceMatrix& dm):
		rowCount(dm.rowCount),
		columnCount(dm.columnCount),
		device(nullptr)
	{
		//rowCountとcolumnCountがともに0でなければ
		//デバイスメモリ確保とデバイスメモリ同士でのコピーを行う
		//rowCountとcolumnCountのどちらかが0の時はdm.device = nullptrのためコピーしない
		unsigned int size = this->rowCount * this->columnCount;
		if(size != 0)
		{
			cudaMalloc((void**)&(this->device), size * sizeof(float));
			cudaMemcpy(this->device, dm.device, size * sizeof(float), cudaMemcpyDeviceToDevice);
		}
	}
	//コピー代入演算子
	DeviceMatrix& operator=(const DeviceMatrix& dm)
	{
		cudaFree(device);
		
		this->rowCount = dm.rowCount;
		this->columnCount = dm.columnCount;
		this->device = nullptr;
		
		//rowCountとcolumnCountがともに0でなければ
		//デバイスメモリ確保とデバイスメモリ同士でのコピーを行う
		//rowCountとcolumnCountのどちらかが0の時はdm.device = nullptrのためコピーしない
		unsigned int size = this->rowCount * this->columnCount;
		if(size != 0)
		{
			cudaMalloc((void**)&(this->device), size * sizeof(float));
			cudaMemcpy(this->device, dm.device, size * sizeof(float), cudaMemcpyDeviceToDevice);
		}
		
		//自分への参照を返す
		return *(this);
	}
	//ムーブコンストラクタ
	DeviceMatrix(DeviceMatrix&& dm):
		rowCount(dm.rowCount),
		columnCount(dm.columnCount),
		device(dm.device)
	{
		//ムーブ元を0行0列に設定
		dm.rowCount = 0;
		dm.columnCount = 0;
		//ムーブ元ポインタをnullptrに設定
		dm.device = nullptr;
	}
	//ムーブ代入演算子
	DeviceMatrix& operator=(DeviceMatrix&& dm)
	{
		//ムーブ先のメモリの開放
		cudaFree(this->device);
		//値とポインタのコピー
		this->rowCount = dm.rowCount;
		this->columnCount = dm.columnCount;
		this->device = dm.device;
		//ムーブ元を0行0列に設定
		dm.rowCount = 0;
		dm.columnCount = 0;
		//ムーブ元ポインタをnullptrに設定
		dm.device = nullptr;
		//自分への参照を返す
		return *(this);
	}
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


