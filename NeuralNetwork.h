#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Random.h"
#include "Utility.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <time.h>

using namespace std;

const double INIT_SIGMA = 5.0e-5;
const double INIT_LAMBDA = 5.0e-7;
const int MAX_EPOCHES = 200000;
const int MAX_TIME = 120;			//120s, 2 min
const double MIN_GRAD = 1e-6;
const int MAX_FAIL_REDUCTION = 10;	// when the number of consecutive zero reduction reach this maximum, stop training
const int SHOW = 100;

class NeuralNetwork
{
private:
	/****************** Virtual Function Definition **************************/
	// Transfer function on hidden unit
	virtual double calcHiddenTrans(double x) = 0;

	// calculate derivative of hidden unit i
	virtual double calcHiddenDerivative(int i) = 0;

	// Transfer function on output unit
	virtual double calcOutputTrans(double x) = 0;

	// calculate gradient upon weight and bias
	virtual double calcGradientDescent(vector<vector<double> > &inputs, vector<double> &expectedOutputs)=0;

	// store address of weights and their gradient into a vector
	virtual void getWeights()=0;
	
	// save network
	virtual void save(fstream &fout)=0;

	/***************** Miscellaneous Function Definition ********************/ 
	double calcDotProduct(double *x, double *y);

protected: 
	// follwing variables can be accessed by all the derived class
		
	// random variable
	Random r;
	
	// number of parameters in the neural network
	int _numOfInput, _numOfHidden, numOfPara;
	
	// hidden unit's input and output
	vector<double>  hInput, hOutput;

	// a vector which store the address of weight factor
	double* *weightAddr;

	// a vector which store the address of gradient descent
	double* *gradientAddr;

public:
	// expected reward
	double expectedReward;

	NeuralNetwork(void);
	~NeuralNetwork(void);
	
	// initialize the network based on the given parameters
	virtual void create(int numOfInput, int numOfHiddenUnit, vector<vector<double> > &s) = 0;
	
	// create network from a file
	virtual void create(string fileName)=0;

	// calculate output of neural network
	virtual double calcOutput(vector<double> &x)=0;

	double scaledConjugateGradient(vector<vector<double> > &inputs, vector<double> &expectedOutputs, double goal, int numOfIteration = MAX_EPOCHES);
};
#endif
