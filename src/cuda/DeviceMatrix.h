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

#include "CuBlasManager.h"

namespace cuda
{

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
			CUDA_CALL(cudaMalloc((void**)&(this->device), size * sizeof(float)));
			CUDA_CALL(cudaMemcpy(this->device, dm.device, size * sizeof(float), cudaMemcpyDeviceToDevice));
		}
	}
	//コピー代入演算子
	DeviceMatrix& operator=(const DeviceMatrix& dm)
	{
		CUDA_CALL(cudaFree(device));
		
		this->rowCount = dm.rowCount;
		this->columnCount = dm.columnCount;
		this->device = nullptr;
		
		//rowCountとcolumnCountがともに0でなければ
		//デバイスメモリ確保とデバイスメモリ同士でのコピーを行う
		//rowCountとcolumnCountのどちらかが0の時はdm.device = nullptrのためコピーしない
		unsigned int size = this->rowCount * this->columnCount;
		if(size != 0)
		{
			CUDA_CALL(cudaMalloc((void**)&(this->device), size * sizeof(float)));
			CUDA_CALL(cudaMemcpy(this->device, dm.device, size * sizeof(float), cudaMemcpyDeviceToDevice));
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
		CUDA_CALL(cudaFree(this->device));
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
			CUDA_CALL(cudaMalloc((void**)&device, rowCount * columnCount * sizeof(float)));
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
		CUDA_CALL_NO_EXCEPTION(cudaFree(device));
	}
	void set(const float* host)
	{
		if(((rowCount * columnCount) == 0))
		{
			return;
		}
		CUBLAS_CALL(cublasSetMatrix(rowCount, columnCount, sizeof(float), host, rowCount, device, rowCount));
	}
	void set(const std::vector<float>& host)
	{
		this->set(host.data());
	}
	void get(float* const host) const
	{
		if(((rowCount * columnCount) == 0))
		{
			return;
		}
		CUBLAS_CALL(cublasGetMatrix(rowCount, columnCount, sizeof(float), device, rowCount, host, rowCount));
	}
	void get(std::vector<float>& host) const
	{
		host.resize(rowCount * columnCount);
		this->get(host.data());
	}
	std::vector<float> get() const
	{
		std::vector<float> host;
		this->get(host);
		return host;
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
	//全成分が引数alphaの行列を取得
	static DeviceMatrix getAlphaMatrix(unsigned int row, unsigned int column, float alpha)
	{
		return DeviceMatrix(row, column, std::vector<float>(row * column, alpha));
	}
	//全成分が0の行列を取得
	static DeviceMatrix get0Matrix(unsigned int row, unsigned int column)
	{
		return DeviceMatrix(row, column, std::vector<float>(row * column, 0.0f));
	}
	//全成分が1の行列を取得
	static DeviceMatrix get1Matrix(unsigned int row, unsigned int column)
	{
		return DeviceMatrix(row, column, std::vector<float>(row * column, 1.0f));
	}
	
};

}

