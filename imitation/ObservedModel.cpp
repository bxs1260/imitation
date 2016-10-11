#include "ObservedModel.h"

ObservedModel::ObservedModel(void) {}
ObservedModel::~ObservedModel(void) {}

ObservedModel::ObservedModel(int numVal, vector<Object> objectsVal, vector<State> statesVal)
: num(numVal), objects(objectsVal), states(statesVal) {}

/*
	Function: toString()
	Desc.	: Returns a String that represents this instance
	Para.	: None
	Return	: string
*/
string ObservedModel::toString() const
{
	size_t i;
	string s;

	// observedModel's number, number of objects and num of states
	s= convertToString(num) + " " + convertToString(objects.size()) + " " + convertToString(states.size()) + "\n";

	// each object's detail
	for (i=0; i<objects.size(); ++i)
		s+= objects[i].toString();
	s+= "\n";

	// each state's detail
	for (i=0; i< states.size(); ++i)
		s += states[i].toString();
	s += "\n";

	return s;
}