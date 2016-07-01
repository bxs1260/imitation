#include "Imitation.h"

using namespace std;

int main(int argc, const char* argv[])
{
	bool debugMode = false;
	int numOfHiddenUnits = 10;
	
	if (argc < 2 || (argv[1] != string("L") && argv[1] != string("T")))
	{
		cout << "Usage: imitation type [numOfHiddenUnits] [debug?]\n" <<
			"type: L, learning; T, testing\n" <<
			"[numOfHiddenUnits]: default is 15\n[debug?]: default is 0" << endl;
		return -1;
	}

	switch (argc)
	{
		case 4:
			numOfHiddenUnits = atoi(argv[2]);
			debugMode = (atoi(argv[3])==1);
			break;	
		case 3:
			numOfHiddenUnits = atoi(argv[2]);
			break;
	}

	Imitation intModel(numOfHiddenUnits, debugMode);
	if (argv[1] == string("L"))
		intModel.learning("observedModel.txt");
	else
	{
		// using default distance function
		intModel.testing();

		// using hand-coded distance function
//		intModel.testing(Imitation::HANDCODE_EW);

//		intModel.testing(Imitation::HANDCODE);
	}

	// generate policy from nn
	//intModel.generatePSFromNN();

	return 0;
}
