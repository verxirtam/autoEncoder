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

#include "Statistics.h"

//入力データの正規化を行う
class Normalization
{
private:
	//平均
	DeviceVector mean;
	//分散共分散行列
	DeviceMatrix varCovMatrix;
	//分散共分散行列の固有値からなるベクトル
	DeviceVector varCovEigenValue;
	//分散共分散行列の固有ベクトルからなる行列
	DeviceMatrix varCovEigenVector;
	//PCA白色化変換行列
	DeviceMatrix pcaWhiteningMatrix;
	//PCA白色化変換行列の逆行列
	DeviceMatrix inversePCAWhiteningMatrix;
	//ZCA白色化変換行列
	DeviceMatrix zcaWhiteningMatrix;
	//ZCA白色化変換行列の逆行列
	DeviceMatrix inverseZCAWhiteningMatrix;

	//成分毎に(-1/2)乗を算出する
	void invSqrtByElement(DeviceVector& W);
	
	//成分毎に(-1)乗を算出する
	void invByElement(DeviceVector& W);
	
	//白色化を行う
	DeviceMatrix getWhitening
		(
			const DeviceMatrix& whiteningMatrix,
			const DeviceMatrix& X,
			const DeviceVector& _1B
		) const;
	//白色化の逆変換を行う
	DeviceMatrix getInverseWhitening
		(
			const DeviceMatrix& inverseWhiteningMatrix,
			const DeviceMatrix& nX,
			const DeviceVector& _1B
		) const;
public:
	//コンストラクタ
	Normalization():
		mean(),
		varCovMatrix(),
		varCovEigenValue(),
		varCovEigenVector(),
		pcaWhiteningMatrix(),
		inversePCAWhiteningMatrix(),
		zcaWhiteningMatrix(),
		inverseZCAWhiteningMatrix()
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
	//分散共分散行列の固有値からなるベクトルを取得
	const DeviceVector& getVarCovEigenValue(void) const
	{
		return varCovEigenValue;
	}
	//分散共分散行列の固有ベクトルからなる行列を取得
	const DeviceMatrix& getVarCovEigenVector(void) const
	{
		return varCovEigenVector;
	}
	//PCA白色化の変換行列を取得する
	const DeviceMatrix& getPCAWhiteningMatrix(void) const
	{
		return pcaWhiteningMatrix;
	}
	//PCA白色化の逆変換行列を取得する
	const DeviceMatrix& getInversePCAWhiteningMatrix(void) const
	{
		return inversePCAWhiteningMatrix;
	}
	//PCA白色化を行う
	DeviceMatrix getPCAWhitening(const DeviceMatrix& X, const DeviceVector& _1B) const
	{
		return getWhitening(pcaWhiteningMatrix, X, _1B);
	}
	//PCA白色化の逆変換を行う
	DeviceMatrix getInversePCAWhitening(const DeviceMatrix& X, const DeviceVector& _1B) const
	{
		return getInverseWhitening(inversePCAWhiteningMatrix, X, _1B);
	}
	//ZCA白色化の変換行列を取得する
	const DeviceMatrix& getZCAWhiteningMatrix(void) const
	{
		return zcaWhiteningMatrix;
	}
	//ZCA白色化の逆変換行列を取得する
	const DeviceMatrix& getInverseZCAWhiteningMatrix(void) const
	{
		return inverseZCAWhiteningMatrix;
	}
	//ZCA白色化を行う
	DeviceMatrix getZCAWhitening(const DeviceMatrix& X, const DeviceVector& _1B) const
	{
		return getWhitening(zcaWhiteningMatrix, X, _1B);
	}
	//ZCA白色化の逆変換を行う
	DeviceMatrix getInverseZCAWhitening(const DeviceMatrix& X, const DeviceVector& _1B) const
	{
		return getInverseWhitening(inverseZCAWhiteningMatrix, X, _1B);
	}
};

