/*
 * =====================================================================================
 *
 *       Filename:  TimeUtil.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年12月03日 08時01分48秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#pragma once

#include <string>

class TimeUtil
{
public:
	//日付文字列からunix timeを求める
	static time_t stringToEpoch(const std::string& timestr);
};


