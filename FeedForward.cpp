#include "FeedForward.h"

FeedForward::FeedForward(void)
{
	// initially the expected reward is zero. It is read from nn.txt if exists
	expectedReward = 0;
	
	// active region for tansig
	/*activeRegion.push_back(-2);
	activeRegion.push_back(2);*/

	// active region for logsig
	activeRegion.push_back(-4);
	activeRegion.push_back(4);
}

FeedForward::~FeedForward(void){}

/*
	Function: calcHiddenTrans()
	Desc	: calculate output of hidden unit
	Para	: x, input
	Return	: double, output of hidden unit
*/
double FeedForward::calcHiddenTrans(double x)
{
	//return 2.0/(1 + exp(-2*x)) - 1;	// tansig

	return 1.0/(1 + exp(-x));			// logsig
}

/*
	Function: calcHiddenDerivative()
	Desc	: calculate derivative of hidden unit i
	Para	: i, the index of hidden unit
	Return	: double, derivative of hidden unit i
*/
double FeedForward::calcHiddenDerivative(int i)
{
	//return 1-pow(hOutput[i], 2);			// tansig

	return hOutput[i] * (1 - hOutput[i]);	// logsig
}

/*
	Function: calcOutputTrans()
	Desc	: calculate output unit's transfer function
	Para	: x, output unit's input parameters
	Return	: double
*/
double FeedForward::calcOutputTrans(double x)
{
	// here purelin used
	return x;
}

/*
	Function: create()
	Desc.	: initialize weight factor
	Para.	: numOfInput, number of input parameters
			  numOfHiddenUnit, number of hidden unit
			  s, sample data which is used to initialize the network
	Return	: None
*/
void FeedForward::create(int numOfInput, int numOfHiddenUnit, vector<vector<double> > &s)
{
	_numOfInput = numOfInput;
	_numOfHidden = numOfHiddenUnit;

	// initialize other vector variables
	initGradient();

	// initialize the network using Nguyen-Widrow algorithm
	//NguyenWidrow(s);
	randomInit();
}

/*
	Function: create`()
	Desc.	: load network from a file.
	Para.	: fielName, the file name where all the parameters are stored.
	Return	: None
	Note	: format,
			  1. expected reward, numOfHidden, numOfInput
			  2. weight factor between input unit and hidden unit
			  3. bias on each hidden unit
			  4  weight factor between hidden unit and output unit, 
			  5. bias for output unit
*/
void FeedForward::create(string fileName)
{
	int i, j;
	double input;

	vector<double> weight;

	fstream fin;
	
	// check whether the fiel exists or not beforehand
	fin.open (fileName.c_str(), ios::in);

	//weight factor of input units
	fin >> expectedReward;
	fin >> _numOfHidden;
	fin >> _numOfInput;

	// clear weight vectors
	iWeights.clear();
	hBias.clear();
	hWeights.clear();

	//weight factor on the link between input unit and hidden units
	for (i=0; i<_numOfHidden; ++i)
	{
		weight.clear();
		for (j=0; j<_numOfInput; ++j)
		{
			fin >> input;
			weight.push_back(input);
		}
		iWeights.push_back(weight);
	}

	// bias on each hidden unit
	for (i=0; i<_numOfHidden; ++i)
	{
		fin >> input;
		hBias.push_back(input);
	}

	// weight factor on the link between hidden unit and output unit
	for (i=0; i<_numOfHidden; ++i)
	{
		fin >> input;
		hWeights.push_back(input);
	}
	// bias on output unit
	fin >> oBias;

	// close file
	fin.close();

	// initialize other vector variables
	initGradient();
}

/*
	Function: calcGradientDescent
	Desc	: calculate gradient descent
	Para	: inputs and expected output
	Return	: mean square error
*/
double FeedForward::calcGradientDescent (vector<vector<double> > &inputs, vector<double> &expectedOutputs)
{
	size_t num, i, j;
	double err, errSum, numFactor;

	// set gradient descent variable to zero
	clearGradient();

	errSum=0;
	numFactor = 2.0/inputs.size();
	// go through each input data
	for (num=0; num<inputs.size(); ++num)
	{		
		// calculate err between the expected output and the actual output
		err = expectedOutputs[num] - calcOutput(inputs[num]);
		errSum += pow(err,2);
		
		// calculate gradient descent of output unit's bias
		oBGradient += -err*numFactor;
		for (i=0; i<iWeights.size(); ++i)			// number of hidden unit
		{
			// calculate gradient descent of weight between hidden unit and output unit
			hWGradient[i] += -err * hOutput[i] * numFactor;

			// calculate gradient descent of hidden unit's bias
			hBGradient[i] += -err*hWeights[i]*calcHiddenDerivative(i)*numFactor;

			// calculate gradient descent of weight between input unit and hidden unit
			for (j=0; j<iWeights[i].size(); ++j)
				iWGradient[i][j] += -err*hWeights[i]*calcHiddenDerivative(i)*inputs[num][j]*numFactor;
		}
	}
	
	return errSum/inputs.size();
}

/*
	Function: getWeights()
	Desc	: store the address of weight factor and gradient descent in a single vector
	Para	: None
	Return	: None
*/
void FeedForward::getWeights()
{
	size_t i, j, k;

	// number of parameters = numOfHidden * (numOfInput+1) + numOfHidden + 1
	numOfPara = _numOfHidden * (_numOfInput + 1) + _numOfHidden + 1;

	weightAddr = new double * [numOfPara];
	gradientAddr = new double * [numOfPara];

	// order: input weight, hidden unit's bias, hidden unit's weight and output unit's bias
	k=0;
	for (i=0; i<iWeights.size(); ++i)	// the last one is bias
	{
		// weight factor between input unit and hidden unit
		for (j=0; j<iWeights[i].size(); ++j)
		{
			weightAddr[k] = &iWeights[i][j];
			gradientAddr[k++] = &iWGradient[i][j];
		}
	}

	for (i=0; i<hBias.size(); ++i)
	{
		// bias
		weightAddr[k] = &hBias[i];
		gradientAddr[k++] = &hBGradient[i];
	}

	// for weight factor between hidden unit and output unit
	for (i=0; i<hWeights.size(); ++i)
	{
		weightAddr[k] = &hWeights[i];
		gradientAddr[k++] = &hWGradient[i];
	}

	// bias on output unit
	weightAddr[k] = &oBias;
	gradientAddr[k++] = &oBGradient;
}

/*
	Function: save()
	Desc.	: save neural network into a file.
	Para.	: fileName, the file name that is used to store the paramters.
	Return	: None
*/
void FeedForward::save (string fileName)
{
	fstream fout;

	fout.open(fileName.c_str(),ios::out);
	
	// write weights through a stream interface
	save(fout);

	fout.close();
}

/*
	Function: save()
	Desc.	: save neural network into a file.
	Para.	: fout, a stream interface that write data to the file
	Return	: None
*/
void FeedForward::save (fstream &fout)
{
	size_t i, j;
	
	// save expected reward, number of hidden unit and input unit
	fout << expectedReward << " " << _numOfHidden << " " << _numOfInput << endl;
	
	//weight factor between input unit and hidden units
	for (i=0; i<iWeights.size(); ++i)
	{
		for (j=0; j<iWeights[i].size(); ++j)
            fout << iWeights[i][j] << " ";
		fout << endl;
	}

	// bias on each hidden unit
	for (i=0; i<iWeights.size(); ++i)
		fout << hBias[i] << " ";
	fout << endl;

	// weight factor between hidden unit and output unit
	for (i=0; i<hWeights.size(); ++i)
		fout << hWeights[i] << " ";
	fout << endl;

	// bias on the output unit
	fout << oBias << endl;
}

/*
	Function: NguyenWidrow()
	Desc	: Nguyen-Widrow initialization algorithm
	Para	: inputs
	Return	: None
	Note	: only the weight factors(Input/Hidden) and bias(Hidden) are initialized with this algorithm, others are randomly initialized
*/
void FeedForward::NguyenWidrow(vector<vector<double> > &s)
{
	int i, j, numOfNonConst, idxOfNonConst;

	// rnd is the value from a uniform random distributation between -1 and 1.
	double rnd, bias, norm, scaleFactor, x, y, sum;

	vector<double> weight, xVector, yVector;

	// minimum and maximum values for each input element
	vector<vector<double> > range;

	// clear all weight factor
	iWeights.clear();
	hBias.clear();
	hWeights.clear();

	// calculate the min and max of input values
	range = minMax(s);

	// how many of input element is non-constant, i.e. min and max is different
	numOfNonConst=0;
	idxOfNonConst=-1;
	for (i=0; i<range.size(); ++i)
	{
		if (range[i][0]!=range[i][1])
		{
			numOfNonConst++;	
			if (idxOfNonConst==-1)
				idxOfNonConst = i;
		}
	}
	scaleFactor = 0.7*pow(1.0*_numOfHidden, 1.0/numOfNonConst);

	// initialize hidden units' weights between -1 and 1
	for (i=0; i<_numOfHidden; ++i)
	{
		norm = 0;
		weight.clear();
		// weight factor on the link between input unit and hidden unit
		for (j=0; j<_numOfInput; ++j)
		{
			if (range[j][0]==range[j][1])
				rnd =0;
			else
			{
				rnd = r.nextDouble(-1,1);
				// calculate the normalization factor of current hidden unit
				norm += rnd*rnd;
			}
			weight.push_back(rnd);
		}

		// adjust the magnitude of weights
		for (j=0; j<_numOfInput; ++j)
			weight[j]*=scaleFactor/sqrt(norm);
		
		iWeights.push_back(weight);

		// bias for each hidden unit
		if (i == _numOfHidden-1)
			bias = 1.0;
		else
			bias = -1 + i*2.0/(_numOfHidden-1);
		
		bias *= scaleFactor * sign<double>(iWeights[i][idxOfNonConst]);
		hBias.push_back(bias);

		// weight factor on the link between hidden unit and output unit
		hWeights.push_back(r.nextDouble(-1,1));
	}

	// bias for the output unit
	oBias = r.nextDouble(-1,1);

	// conversion of net inputs of [-1, 1] to [activeMin, activeMax], [-2, 2] for tansig and [-4, 4] for logsig
	x = 0.5*(activeRegion[1] - activeRegion[0]);
	y = 0.5*(activeRegion[1] + activeRegion[0]);

	for (i=0; i<_numOfHidden; ++i)
	{
		for (j=0; j<_numOfInput; ++j)
			iWeights[i][j]*=x;

		hBias[i]=x*hBias[i]+y;
	}

	// conversion of inputs of PR to [-1, 1]	
	for (i=0; i<_numOfInput; ++i)
	{
		if (range[i][0] == range[i][1]) 
		{
			x=0.0;
			y=0.0;
		}
		else
		{
			x = 2.0 / (range[i][1]-range[i][0]);
			y=1-range[i][1]*x;
		}

		xVector.push_back(x);
		yVector.push_back(y);
	}

	for (i=0; i<_numOfHidden; ++i)
	{		
		sum=0;
		for (j=0; j<_numOfInput; ++j)
		{
			sum += iWeights[i][j]*yVector[j];
			iWeights[i][j]*=xVector[j];
		}

		hBias[i] += sum;
	}
}

/*
	Function: minMax
	Desc	: calculate the ranges of each column in the given matrix
	Para	: M*N matrix
	Return	: N*2 matrix which including minimum and maximum values for each column of M.
*/
vector<vector<double> > FeedForward::minMax(vector<vector<double> > &v)
{
	size_t i;

	vector<double> r;
	vector<vector<double> > m, t;

	// transpose
	t = transpose(v);
	for (i=0; i<t.size(); ++i)
	{
		r.clear();

		// sort in each input element
		sort(t[i].begin(), t[i].end());

		// add min and max
		r.push_back(t[i].front());
		r.push_back(t[i].back());

		m.push_back(r);
	}
	
	return m;
}

/*
	Function: transpose
	Desc	: return transpose matrix
	Para	: M*N matrix
	Return	: N*M matrix
*/
vector<vector<double> > FeedForward::transpose(vector<vector<double> > &v)
{
	size_t i, j;

	vector<double> r;

	// transpose matrix
	vector<vector<double> > t;
	
	for (j=0; j<v[0].size(); ++j)
	{
		r.clear();
		// for one column
		for (i=0; i<v.size(); ++i)
			r.push_back(v[i][j]);

		t.push_back(r);
	}

	return t;
}

/*
	Function: calcOutput()
	Desc	: calculate network's output
	Para	: x, input vector
	Return	: double,
*/
double FeedForward::calcOutput(vector<double> &x)
{
	size_t i, j;
	double inputSum, outputSum;
	
	outputSum=0;
	for (i=0; i<hWeights.size(); ++i)
	{
		// first, calculate weighted sum of input parameter
		inputSum=0;
		for (j=0; j<iWeights[i].size(); ++j)
			inputSum += x[j]*iWeights[i][j];

		hInput[i] = hBias[i] + inputSum;

		// second, calculate output for each hidden unit
		hOutput[i] = calcHiddenTrans(hInput[i]);
	
		outputSum += hOutput[i] * hWeights[i];
	}
	outputSum += oBias;

	//Test(x,outputSum);
	return calcOutputTrans(outputSum);
}

/*
	Function: clearGradient()
	Desc	: set the gradient descent variables to 0
	Para	: None
	Return	: None
*/
void FeedForward::clearGradient()
{
	size_t i, j;

	for (i=0; i<iWGradient.size(); ++i)			// number of hidden unit
	{
		for (j=0; j<iWGradient[i].size(); ++j)	// number of input parameter
			// clear gradient of hidden unit's weight factor
			iWGradient[i][j]=0;

		// clear gradient descent of hidden unit's bias
		hBGradient[i]=0;

		// clear gradient descent of output unit's weight factor
		hWGradient[i]=0;
	}

	// clear gradient descent of output unit's bias
	oBGradient = 0;
}

/*
	Function: initGradient()
	Desc	: initialize some variable with vector<double> type
	Para	: None
	Return	: None
*/
void FeedForward::initGradient()
{	
	vector<double> w;

	// clear vector variable
	iWGradient.clear();
	hBGradient.clear();
	hWGradient.clear();
	hInput.clear();
	hOutput.clear();

	// gradient variable
	w.insert(w.end(), _numOfInput, 0.0);
	iWGradient.insert(iWGradient.end(), _numOfHidden, w);

	hBGradient.insert(hBGradient.end(), _numOfHidden, 0.0);
	hWGradient.insert(hWGradient.end(), _numOfHidden, 0.0);
	
	// input/output variable
	hInput.insert(hInput.end(), _numOfHidden, 0.0);
	hOutput.insert(hOutput.end(), _numOfHidden, 0.0);
}

void FeedForward::Test(vector<double> &input, double output)
{
	size_t i;
	fstream fout;

	fout.open("o_inout.txt", ios::out);

	// output input parameter
	fout << "input: " << endl;
	for (i=0; i<input.size(); ++i)
		fout << input[i] << " ";
	fout << output << endl;

	fout << "h input: " << endl;
	// output hidden unit's input
	for (i=0; i<hInput.size(); ++i)
		fout << hInput[i] << " ";
	fout << endl;

	fout << "h output: " << endl;
	// output hidden unit's output
	for (i=0; i<hOutput.size(); ++i)
		fout << hOutput[i] << " ";
	fout << endl;

	fout.close();
}

void FeedForward::randomInit()
{
	int i, j;
	vector<double> w;

	iWeights.clear();
	hBias.clear();
	hWeights.clear();

double rnd, norm;
	for (i=0; i<_numOfHidden; ++i)
	{
		
		norm = 0;
		w.clear();
		// weight factor on the link between input unit and hidden unit
		for (j=0; j<_numOfInput; ++j)
		{
			rnd = r.nextDouble(-1,1);
			norm += rnd*rnd;	// calculate the normalization factor of current hidden unit

			w.push_back(rnd);
		}

		// adjust the magnitude of weights
		for (j=0; j<_numOfInput; ++j)
			w[j]/=sqrt(norm);

		iWeights.push_back(w);

		// bias for hidden unit
		hBias.push_back(r.nextDouble(-1,1));

		// weight factor on the link between hidden unit and output unit
		hWeights.push_back(r.nextDouble(-1,1));
	}

	// bias for the output unit
	oBias = r.nextDouble(-1,1);
}