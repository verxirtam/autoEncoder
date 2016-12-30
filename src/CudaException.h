/*
 * =====================================================================================
 *
 *       Filename:  CudaException.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月31日 01時58分45秒
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

class CudaException : public std::runtime_error
{
public:
	CudaException(const std::string& what_arg):
		runtime_error(what_arg.c_str())
	{
	}
	CudaException(const char* what_arg):
		runtime_error(what_arg)
	{
	}
};


