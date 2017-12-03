/*
 * =====================================================================================
 *
 *       Filename:  TimeUtil.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年12月03日 09時12分03秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <time.h>

#include "TimeUtil.h"


//日付文字列からunix timeを求める
time_t TimeUtil::stringToEpoch(const std::string& timestr_jst)
{
	std::string timestr = timestr_jst;
	
	::tm t;
	
	::strptime(timestr_jst.c_str(), "%Y/%m/%d %H:%M:%S", &t);
	
	return ::mktime(&t);
}

