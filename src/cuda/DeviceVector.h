/*
 * =====================================================================================
 *
 *       Filename:  DeviceVector.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月19日 02時36分49秒
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
#include <initializer_list>

#include "CuBlasManager.h"

class DeviceVector
{
private:
	//次元
	unsigned int dimension;
	//デバイスメモリのアドレス
	float* device;
public:
	//コピーコンストラクタ
	DeviceVector(const DeviceVector& dv):
		dimension(dv.dimension),
		device(nullptr)
	{
		//dimensionが0でなければ
		//デバイスメモリ確保とデバイスメモリ同士でのコピーを行う
		//dimensionが0の時はdv.device = nullptrのためコピーしない
		if(this->dimension != 0)
		{
			CUDA_CALL(cudaMalloc((void**)&(this->device), this->dimension * sizeof(float)));
			CUDA_CALL(cudaMemcpy(this->device, dv.device, this->dimension * sizeof(float), cudaMemcpyDeviceToDevice));
		}
	}
	//コピー代入演算子
	DeviceVector& operator=(const DeviceVector& dv)
	{
		
		//dimensionが0でなければ
		//デバイスメモリ確保とデバイスメモリ同士でのコピーを行う
		//dimensionが0の時はdv.device = nullptrのためコピーしない
		if(dv.dimension != 0)
		{
			if(this->dimension != dv.dimension)
			{
				//コピー先のメモリ解放
				CUDA_CALL(cudaFree(this->device));
				this->device = nullptr;
				//コピー先のメモリ確保
				CUDA_CALL(cudaMalloc((void**)&(this->device), dv.dimension * sizeof(float)));
			}
			CUDA_CALL(cudaMemcpy(this->device, dv.device, dv.dimension * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		else
		{
			//dv.dimension == 0 の場合
			//コピー先のメモリ解放
			CUDA_CALL(cudaFree(this->device));
			this->device = nullptr;
		}
		
		//次元のコピー
		this->dimension = dv.dimension;
		
		//自身への参照を返す
		return *this;
	}
	//ムーブコンストラクタ
	//dvのアドレスを付け替える
	DeviceVector(DeviceVector&& dv):
		dimension(dv.dimension),
		device(dv.device)
	{
		//ムーブ元はdimension = 0にする
		dv.dimension = 0;
		//ムーブ元はnullptrにする
		dv.device = nullptr;
	}
	//ムーブ代入演算子
	//dvのアドレスを付け替える
	DeviceVector& operator=(DeviceVector&& dv)
	{
		//次元のコピー
		this->dimension = dv.dimension;
		
		//ムーブ先のメモリ解放
		CUDA_CALL(cudaFree(this->device));
		//ムーブするアドレスの付け替え
		this->device = dv.device;
		//ムーブ元はdimension = 0にする
		dv.dimension = 0;
		//ムーブ元のアドレスをnullptrにする
		dv.device = nullptr;
		//自身への参照を返す
		return *this;
	}
	//コンストラクタ
	//(デフォルトコンストラクタにもなっている)
	DeviceVector(unsigned int d = 0):
		dimension(d),
		device(nullptr)
	{
		if(d != 0)
		{
			CUDA_CALL(cudaMalloc((void**)&device, dimension * sizeof(float)));
		}
	}
	//コンストラクタ
	DeviceVector(const std::vector<float>& v):
		DeviceVector(v.size())
	{
		this->set(v.data());
	}
	//コンストラクタ
	DeviceVector(const std::initializer_list<float> v):
		DeviceVector(std::vector<float>(v))
	{
	}
	//デストラクタ
	~DeviceVector()
	{
		CUDA_CALL(cudaFree(device));
	}
	//値の設定
	//配列のサイズがdimensionになっていることが前提
	//範囲チェックはしない
	void set(const float* const host)
	{
		CUBLAS_CALL(cublasSetVector(dimension, sizeof(float), host, 1, device, 1));
	}
	//値の設定(vector版)
	void set(const std::vector<float>& host)
	{
		this->set(host.data());
	}
	//値の取得
	//配列のサイズがdimensionになっていることが前提
	//範囲チェックはしない
	void get(float* const host) const
	{
		CUBLAS_CALL(cublasGetVector(dimension, sizeof(float), device, 1, host, 1));
	}
	//値の取得(vector版)
	void get(std::vector<float>& host) const
	{
		if(this->dimension != host.size())
		{
			//サイズをDeviceVectorのdimensionに合わせる
			host.resize(this->dimension);
		}
		//値の設定
		this->get(host.data());
	}
	std::vector<float> get() const
	{
		std::vector<float> host;
		this->get(host);
		//返り値はムーブ代入される
		return host;
	}
	//デバイスメモリのアドレスを取得
	float* getAddress() const
	{
		return device;
	}
	//次元を取得
	unsigned int getDimension() const
	{
		return dimension;
	}
	//全成分が引数alphaのベクトルを取得
	static DeviceVector getAlphaVector(unsigned int dimension, float alpha)
	{
		return DeviceVector(std::vector<float>(dimension, alpha));
	}
	//全成分が0のベクトルを取得
	static DeviceVector get0Vector(unsigned int dimension)
	{
		return getAlphaVector(dimension, 0.0f);
	}
	//全成分が1のベクトルを取得
	static DeviceVector get1Vector(unsigned int dimension)
	{
		return getAlphaVector(dimension, 1.0f);
	}
};


