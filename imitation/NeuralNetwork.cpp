#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(void) {}
NeuralNetwork::~NeuralNetwork(void) {}

/*
	Function: scaledConjugateGradient()
	Desc	: Scaled Conjugate Gradient algorithm
	Para	: inputs, input parameter
			  expectedOutputs, expected output
			  goal, performance goal
	Return	: None
*/
double NeuralNetwork::scaledConjugateGradient(vector<vector<double> > &inputs, vector<double> &expectedOutputs, double goal, int numOfIteration)
{
	int i, j, epoch;

	// determines change in weight for second derivative approximation
	double sigma, sigmaBase, alpha, beta, delta, mu, lambda, lambdaRaised, deltaK;
	
	double err, oldErr, normR, normSqrP, oldNormSqrP, dotProductOfRR, dotProductOfRPreR;

	// direction
	double *oldWeight, *oldGradient, *p, *r, *s;

	fstream ferr;	//, fhinput, foutput, fnn;

	// update detail in each epoch
	ferr.open("n_scg.txt", ios::out);

	//// network configuration in each epoch
	//fnn.open("n_nn.txt", ios::out);

	//// input for each hidden unit
	//fhinput.open("n_hinput.txt", ios::out);

	//// current/expected output and difference for each input
	//foutput.open("n_output.txt", ios::out);
	//foutput << "curr: " << setw(10) << "expected: " << setw(10) << "diff" << endl;

	time_t startTime = time(NULL);

	// store the address of all the weight factors and gradient descent into a vector
	getWeights();

	oldWeight = new double[numOfPara];
	oldGradient = new double[numOfPara];
	p = new double[numOfPara];
	r = new double[numOfPara];
	s = new double[numOfPara];

	ferr << "START\n";
	// 1. initialization
	bool success = true;
	sigmaBase = INIT_SIGMA;

	// regulating the indefiniteness of the Hessian
	lambdaRaised = 0;
	lambda = INIT_LAMBDA;

	// the following gradient descent is calculated on all the sample data
	err=calcGradientDescent(inputs, expectedOutputs);
	
	// initially the direction is point to negative gradient descient
	normR = 0;
	for (i=0; i<numOfPara; ++i)
	{
		p[i]=- *gradientAddr[i];
		r[i]=p[i];
	
		// calculate l2-Norm of p, r
		normR += r[i]*r[i];
	}
	// initially l2-Norm of P is same as R's
	normSqrP = normR;
	oldNormSqrP = normSqrP;
	normR = sqrt(normR);	
	
	string msg = "";
	/*
		iCount is used to indicate how many consecutive zero-reduction happen
			- increase when there is zero-reduction
			- reset to 0 when there is non-zero reduction
			- when it reach MAX_FAIL_REDUCTION, stop training process
	*/
	int iCount=0;
	double distance;
	for (epoch=0; epoch<=numOfIteration; ++epoch)
	{
		/*fnn << "round: " << epoch << endl;
		fhinput << "round: " << epoch << endl;
		foutput << "round: " << epoch << endl;*/
		
		//// output network configuration
		//save(fnn);	

		// output hidden units' inputs
		for (i=0; i<inputs.size(); ++i)
		{	
			distance = calcOutput(inputs[i]);

			//foutput << distance << setw(10) << expectedOutputs[i] << setw(10)  << expectedOutputs[i]-distance << endl;
			//// output each hidden units' input
			//for (j=0; j<_numOfHidden; ++j)
			//	fhinput << hInput[j] << " ";
			//fhinput << endl;
		}
		//foutput << endl;

		// stop criteria
		if (err <= goal)
			msg = "Performance goal is satisfied.";
		else if (normR < MIN_GRAD)
			msg = "Minimum gradient reached, performance goal was not satisfied.";
		else if (time(NULL)-startTime > MAX_TIME)
			msg = "Maximum time elapsed, performance goal was not satisfied.";
		//else if (epoch==MAX_EPOCHES)
		else if (epoch==numOfIteration)
			msg = "Maximum epoch reached, performance goal was not satisfied!";
		else if (iCount == MAX_FAIL_REDUCTION)
			msg = "Maximum zero-reduction reached, performance goal was not satisfied!";

		if (!msg.empty())
			cout << "epoch " << epoch << "/" << MAX_EPOCHES << " err: " << err << "/" << goal << " grad(R) " << normR << "/" << MIN_GRAD << " normSqrP " << normSqrP << endl;

		if (epoch%SHOW==0 || !msg.empty() || !success)
			ferr << endl << "epoch " << epoch << "/" << MAX_EPOCHES << " err: " << err << "/" << goal << " grad(R) " << normR << "/" << MIN_GRAD << " normSqrP " << normSqrP << endl;

		// exit loop when stop criteria is satisfied
		if (!msg.empty())
			break;

		// store current gradient descent and weight factor
		for (i=0; i<numOfPara; ++i)
		{
			oldGradient[i]= *gradientAddr[i];
			oldWeight[i]= *weightAddr[i];
		}	
		oldErr=err;

		// 2. if success = true, then calculate second order information
		if (success)
		{
			if (epoch%SHOW==0)
				ferr << "SUCCESS" << endl;
			
			//2.1
			sigma = sigmaBase/sqrt(normSqrP);

			// calculate new weight base on p and sigma
			for (i=0; i<numOfPara; ++i)
				*weightAddr[i] += sigma * p[i];

			// calculate gradient descent (first order deviation)
			err = calcGradientDescent(inputs, expectedOutputs);

			delta = 0;
			for (i=0; i<numOfPara; ++i)
			{
				s[i]=(*gradientAddr[i]-oldGradient[i])/sigma;
				delta += p[i]*s[i];
			}

			if (epoch%SHOW==0)
			{
				ferr << "delta = " << delta << endl;
				ferr << "END OF SUCCESS true" << endl;
			}
		}

		// 3. scale delta
		delta += (lambda-lambdaRaised)*normSqrP;
		if (epoch%SHOW==0 || !success)
			ferr << "DELTA SCALED = " << delta << " lambda " << lambda << " lambdaRaised " << lambdaRaised << endl;
		
		// 4. if delta <=0, make the Hessian matrix positive difinite
		if (delta<=0)
		{
			lambdaRaised = 2*(lambda-delta/normSqrP);
			delta = -delta + lambda * normSqrP;
			lambda = lambdaRaised;

			if (epoch%SHOW==0 || !success)
				ferr << "hessian: " << lambdaRaised << " delta= " << delta << " lambda= " << lambda << endl;
		}

		// 5. calcualte step size
		mu = calcDotProduct(p,r);
		alpha = mu/delta;

		if (epoch%SHOW==0 || !success)
			ferr << "mu= " << mu << " alpha= " << alpha << endl;

		// 6. calculate the comparison parameter
		// change weight factor first, if not accept, roll back
		for (i=0; i<numOfPara; ++i)
			*weightAddr[i] = oldWeight[i] + alpha*p[i];

		// calculate gradient descent, main purpose is to calculate err
		err = calcGradientDescent(inputs, expectedOutputs);

		// may need calculate err beforehand
		deltaK = 2.0*delta*(oldErr - err)/pow(mu,2);
		if (epoch%SHOW==0 || !success)
			ferr << "delta K= " << deltaK << endl;

		// 7. if delta K >=0, a successful reduction in error can be made:
		if (deltaK >=0)
		{
			if (deltaK ==0)
				iCount++;
			else
				iCount=0;

			// the change on the weight factor already be made in previous step
			if (epoch%SHOW==0 || !success)
				ferr << "ERROR reduction of " << (oldErr-err)/oldErr*100.0 << endl;
			
			dotProductOfRR=0;
			dotProductOfRPreR = 0;
			for (i=0; i<numOfPara; ++i)
			{
				// calculate the dot product of previous r and current one.
				dotProductOfRPreR += - *gradientAddr[i] * r[i];

				// update r and calculate dot product of r
				r[i]=- *gradientAddr[i];
				dotProductOfRR += r[i]*r[i];
			}
			normR = sqrt(dotProductOfRR);

			lambdaRaised=0;
			success = true;

			// if epoch > number of parameter, restart scaled conjugate gradient
			if (epoch%numOfPara ==0) 
			{
				// restart = true;
				for (i=0; i<numOfPara; ++i)
					p[i]=r[i];

				normSqrP = dotProductOfRR;
			}	
			else
			{	
				beta = (dotProductOfRR - dotProductOfRPreR)/mu;
				normSqrP = 0;
				for (i=0; i<numOfPara; ++i)
				{
                    p[i] = r[i] + beta*p[i];
					normSqrP += p[i]*p[i];
				}
			}

			// if delta K >=0.75, reduce the scale parameter
			if (deltaK >= 0.75)
				lambda = 0.25*lambda;
		}	
		else
		{
			if (epoch%SHOW==0 || !success)
				ferr << "no reduction!" << endl;
			
			// undo the change made in previous step (step 6)
			for (i=0; i<numOfPara; ++i)
				*weightAddr[i] = oldWeight[i];
			err=oldErr;

			lambdaRaised = lambda;
			success = false;
		}

		// 8. if delta K < 0.25, increase the scale parameter
		if (deltaK < 0.25)
			lambda += delta * (1-deltaK)/oldNormSqrP;

		oldNormSqrP = normSqrP;
	}

	delete [] weightAddr;
	delete [] gradientAddr;

	delete [] oldWeight;
	delete [] oldGradient;
	delete [] p;
	delete [] r;
	delete [] s;

	cout << msg.c_str() << endl;
	ferr << msg.c_str() << endl;
	ferr.close();

	//fhinput.close();
	//foutput.close();
	//fnn.close();
	return err;
}

/*
	Function: calcDotProduct
	Desc	: calculate dot product of two vector
	Param	: vectors x and y
	Return	: double, return -1 if two vector have different size.
*/
double NeuralNetwork::calcDotProduct(double *x, double *y)
{
	double sum = 0;

	for (int i=0; i<numOfPara; ++i)
		sum += x[i]*y[i];

	return sum;
}
