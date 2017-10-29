#pragma once

//2変数関数Func2_1の第2変数にFunc1_1を合成する

template<class Func2_1, class Func1_1>
class Composite1st2_1
{
public:
	//関数
	__host__ __device__
	static float apply(float x, float y)
	{
		return Func2_1::apply(Func1_1::apply(x), y);
	}
};


//2変数関数Func2_1の第2変数にFunc1_1を合成する

template<class Func2_1, class Func1_1>
class Composite2nd2_1
{
public:
	//関数
	__host__ __device__
	static float apply(float x, float y)
	{
		return Func2_1::apply(x, Func1_1::apply(y));
	}
};


template<class Func1_1>
class ApplyDiff1_1
{
	//関数
	__host__ __device__
	static float apply(float x)
	{
		return Func1_1::applyDiff(x);
	}
};





