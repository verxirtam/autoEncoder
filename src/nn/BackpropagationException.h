/*
 * =====================================================================================
 *
 *       Filename:  BackpropagationException.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2016年12月31日 02時17分59秒
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

class BackpropagationException : public std::runtime_error
{
public:
	BackpropagationException(const std::string& what_arg):
		runtime_error(what_arg.c_str())
	{
	}
	BackpropagationException(const char* what_arg):
		runtime_error(what_arg)
	{
	}
};
