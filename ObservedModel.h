#ifndef OBSERVEDMODEL_H
#define OBSERVEDMODEL_H

#include <string>

#include "State.h"
#include "Object.h"
#include "Utility.h"

using namespace std;
class ObservedModel
{
public:
	int num;					// demonstration's number
	vector<Object> objects;		// objects in this instance
	vector<State> states;		// state in this instance

	ObservedModel(void);
	ObservedModel(int numVal, vector<Object> objectsVal, vector<State> statesVal);

	~ObservedModel(void);
	
	// return a string representing this instance
	string toString() const;
};

#endif
