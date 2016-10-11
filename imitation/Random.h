#ifndef RANDOM_H
#define RANDOM_H

#include <ctime>
#include <cmath>
#include <fstream>

using namespace std;

class Random
{
private:
	bool hasNextNextGaussian;
	double nextNextGaussian;
public:
	Random(void);
	~Random(void);

	// generate a random double value in [0,1)	
	double nextDouble();

	// generate a random double value in [0,upper)
	double nextDouble(int upper);

	// generate a random double value in [lower,upper)
	double nextDouble(int lower, int upper);

	// generate a random integer in [0, upper)
	int nextInt(int upper);

	// generate the next pseudorandom, uniformly distributed double value between 0.0 and 1.0 from this random number generator's sequence. 
	double nextGaussian();

	// generate the next random double base on given mean and standard deviation 
	double nextGaussian(double mean, double stdev);
};

#endif
