/*
 * =====================================================================================
 *
 *       Filename:  DeviceVectorUtils.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年10月03日 00時36分34秒
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

#include <fstream>

namespace cuda
{

float compare(const DeviceMatrix& m0, const DeviceMatrix& m1);
float compare(const DeviceVector& v0, const DeviceVector& v1);

DeviceVector& readFromCsvFile(const std::string& csvFileName, DeviceVector& deviceVector);

DeviceMatrix& readFromCsvFile(const std::string& csvFileName, DeviceMatrix& deviceMatrix);

void writeToCsvFile(const std::string& csvFileName, const DeviceVector& deviceVector);

void writeToCsvFile(const std::string& csvFileName, const DeviceMatrix& deviceMatrix);

}

