#include "Random.h"

Random::Random(void)
{
	hasNextNextGaussian = false;
	srand((unsigned)time(0));
}

Random::~Random(void) {}

/*
	Function: nextDouble()
	Desc.	: Generate a random double value from 0 to 1
	Para.	: None
	Return	: a random value in [0,1)
*/
double Random::nextDouble()
{
	//return a random value in the range [0,1);
	return (double)rand()/(RAND_MAX+1.0);
}

/*
	Function: nextDouble()
	Desc.	: Generate a random double value from 0 to a specified upper bound.
	Para.	: upper, a upper bound of the range
	Return	: a random value in [0,upper)
*/
double Random::nextDouble(int upper)
{
	// return a random value in the range [0,upper)
	return nextDouble() * upper;
}

/*
	Function: nextDouble()
	Desc.	: Generate a random double value in a specified range
	Para.	: lower, lower bound of the range.
			  upper, upper bound of the range.
	Return	: a random value in [lower, upper)
*/
double Random::nextDouble(int lower, int upper)
{
	return (upper-lower)*nextDouble() - upper;
}

/*
	Function: nextGaussian()
	Desc.	: Generate the next pseudorandom, uniformly distributed double value between 0.0 and 1.0 from this random number generator's sequence. 
	Para.	: None
	Return	: The next pseudorandom, uniformly distributed double value between 0.0 and 1.0 from this random number generator's sequence. 
*/
double Random::nextGaussian ()
{
	double u, v, x, multiplier;

	if (hasNextNextGaussian)
	{
		hasNextNextGaussian = false;
		return nextNextGaussian;
	} else{
		do 
		{
			u=2.0*nextDouble() - 1.0;	//between -1.0 and 1.0
			v=2.0*nextDouble() - 1.0;	//between -1.0 and 1.0
			x = u*u + v*v;
		} while (x>=1.0 || x==0);
		
		multiplier = sqrt(-2.0*log(x)/x);
		nextNextGaussian = v*multiplier;
		hasNextNextGaussian = true;
		
		return u*multiplier;
	}
}

/*
	Function: nextGaussian
	Desc.	: Base on mean and standard deviation, randomly generate a new value
	Para.	: Gaussian distribution which representing the distribution of the parameter
				- mean, mean of probability distribution
				- stdev, standard deviation of probability distribution 
	Return	: new value of one parameter in the network
*/
double Random::nextGaussian(double mean, double stdev)
{
	// chose from (approximately) the usual normal distribution with mean 0.0 and standard deviation 1.0
	double nextValue = nextGaussian();

	// convert into random number within the range specified by mean and standard deviation
	return (nextValue*stdev+mean);
}

/*
	Function: nextInt()
	Desc.	: Generate a random integer from 0 to a specified upper bound.
	Para.	: upper, a upper bound of the range
	Return	: a random value in [0,upper)
*/
int Random::nextInt(int upper)
{
	// return a random integer in the range [0,upper)
	return (int)(nextDouble(upper));
}