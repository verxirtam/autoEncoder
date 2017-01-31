/*
 * =====================================================================================
 *
 *       Filename:  Normalization.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年01月29日 15時19分32秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include "CuBlasFunction.h"
#include "CuSolverDnFunction.h"

//入力データの正規化を行う
class Normalization
{
private:
	//平均
	DeviceVector mean;
	//分散共分散行列
	DeviceMatrix varCovMatrix;
	//PCA白色化変換行列
	DeviceMatrix pcaWhiteningMatrix;
	//ZCA白色化変換行列
	DeviceMatrix zcaWhiteningMatrix;

	//成分毎に(-1/2)乗を算出する
	void invSqrtByElement(DeviceVector& W);
	
	//白色化を行う
	DeviceMatrix getWhitening(const DeviceMatrix& whiteningMatrix, const DeviceMatrix& X) const;
public:
	//コンストラクタ
	Normalization():
		mean(),
		varCovMatrix(),
		pcaWhiteningMatrix(),
		zcaWhiteningMatrix()
	{
		
	}
	//正規化用のデータを元に平均・白色化の変換行列を作成する
	void init(const DeviceMatrix& X);
	//平均を取得
	const DeviceVector& getMean(void) const
	{
		return mean;
	}
	//分散共分散行列を取得する
	const DeviceMatrix& getVarCovMatrix(void) const
	{
		return varCovMatrix;
	}
	//PCA白色化の変換行列を取得する
	const DeviceMatrix& getPCAWhiteningMatrix(void) const
	{
		return pcaWhiteningMatrix;
	}
	//PCA白色化を行う
	DeviceMatrix getPCAWhitening(const DeviceMatrix& X) const
	{
		return getWhitening(pcaWhiteningMatrix, X);
	}
	//ZCA白色化の変換行列を取得する
	const DeviceMatrix& getZCAWhiteningMatrix(void) const
	{
		return zcaWhiteningMatrix;
	}
	//ZCA白色化を行う
	DeviceMatrix getZCAWhitening(const DeviceMatrix& X) const
	{
		return getWhitening(zcaWhiteningMatrix, X);
	}
};

