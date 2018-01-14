#pragma once

//2変数関数Func2to1の第1変数にFunc1to1を合成する
template<class Func2to1, class Func1to1>
class Func2to1Composite1st
{
public:
	//関数
	__host__ __device__
	static float apply(float x, float y)
	{
		return Func2to1::apply(Func1to1::apply(x), y);
	}
};


//2変数関数Func2to1の第2変数にFunc1to1を合成する
template<class Func2to1, class Func1to1>
class Func2to1Composite2nd
{
public:
	//関数
	__host__ __device__
	static float apply(float x, float y)
	{
		return Func2to1::apply(x, Func1to1::apply(y));
	}
};

//微分を適用する
template<class Func1to1>
class Func1to1ApplyDiff
{
public:
	//関数
	__host__ __device__
	static float apply(float x)
	{
		return Func1to1::applyDiff(x);
	}
};





