/*
 * =====================================================================================
 *
 *       Filename:  CuBlasException.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月31日 02時13分43秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include <stdexcept>

#include <string>

namespace cuda
{

class CuBlasException : public std::runtime_error
{
public:
	CuBlasException(const std::string& what_arg):
		runtime_error(what_arg.c_str())
	{
	}
	CuBlasException(const char* what_arg):
		runtime_error(what_arg)
	{
	}
};

}

