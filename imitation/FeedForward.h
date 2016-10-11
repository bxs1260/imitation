#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "NeuralNetwork.h"
#include "Utility.h"

#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

using namespace std;

class FeedForward : public NeuralNetwork
{
	// active region of hidden unit's transfer function
	vector<double> activeRegion;

	// the weight factors and their gradient on the link between input unit and hidden unit
	// M*N, M represents number of hidden unit, N represents number of input element,
	// each row represent all the weight factor for one single hidden unit
	vector<vector<double> > iWeights, iWGradient;
	
	// the bias and their gradient on each hidden unit
	vector<double> hBias, hBGradient;

	// the weight factors and their gradient on the link between hidden unit and output unit
	vector<double> hWeights, hWGradient;
	
	// the bias and its gradient on the output unit
	double oBias, oBGradient;

	/***************** Overwrite Virtual Function ****************************/
	// Transfer function on hidden unit
	double calcHiddenTrans(double x);
	
	// calculate derivative of hidden unit i
	double calcHiddenDerivative(int i);

	// Transfer function on output unit
	double calcOutputTrans(double x);

	// calculate gradient upon weight and bias
	double calcGradientDescent(vector<vector<double> > &inputs, vector<double> &expectedOutputs);

	// store address of weights and their gradient into a vector
	void getWeights();

	// save network
	void save(fstream &fout);

	/***************** Miscellaneous Function Definition ********************/ 
	// clear gradient descent
	void clearGradient();

	// initialization vector variable
	void initGradient();

	// calculate minimum and maximum of each column in the given matrix
	vector<vector<double> > minMax(vector<vector<double> > &v);

	// Nguyen-Widrow initialization
	void NguyenWidrow(vector<vector<double> > &s);

	// return transpose matrix for the given matrix
	vector<vector<double> > transpose(vector<vector<double> > &v);

	// test purpose
	void Test(vector<double> &input, double output);
	
	void randomInit();

public:
	FeedForward(void);
	~FeedForward(void);

	// initialize the network based on the given parameters
	void create(int numOfInput, int numOfHiddenUnit, vector<vector<double> > &s);

	// create network from a file
	void create(string fileName);

	// calculate output of neural network
	double calcOutput(vector<double> &x);

	// save network
	void save(string fileName);
};
#endif
