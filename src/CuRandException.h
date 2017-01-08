/*
 * =====================================================================================
 *
 *       Filename:  CuRandException.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年01月08日 16時57分30秒
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

class CuRandException : public std::runtime_error
{
public:
	CuRandException(const std::string& what_arg):
		runtime_error(what_arg.c_str())
	{
	}
	CuRandException(const char* what_arg):
		runtime_error(what_arg)
	{
	}
};
