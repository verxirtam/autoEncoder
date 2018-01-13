/*
 * =====================================================================================
 *
 *       Filename:  CuSolverDnFunction.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年01月22日 22時36分04秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include "DeviceVector.h"
#include "DeviceMatrix.h"

#include "CuSolverDnManager.h"

//対称行列に対する固有値・固有ベクトルを求める
// A * V = V * diag(W)
//A 対称行列
//W Aの固有値からなるベクトル(成分は固有値の昇順)
//V Aの固有ベクトル
void DnSsyevd
	(
		const DeviceMatrix& A,
		DeviceVector& W,
		DeviceMatrix& V
	);

