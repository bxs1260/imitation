#include "InternalState.h"

InternalState::InternalState(void) : action(-1), extStateNum(-1), distance(-1){}

InternalState::InternalState(int actionVal, State stateVal, int extStateNumVal, double distanceVal) 
	: action(actionVal), state(stateVal), extStateNum(extStateNumVal), distance(distanceVal) 
{
	// update nextTo
}

InternalState::~InternalState(void) {}

/*
	Function: clear()
	Desc.	: clear this instance
	Para.	: None
	Return	: None
*/
void InternalState::clear ()
{
	action=-1;
	extStateNum = -1;
	distance = 0;

	state.clear();
}

bool InternalState::operator==(InternalState s)
{
	// compare state and correpsonding observed state
	return (state == s.state && extStateNum == s.extStateNum);
}

bool InternalState::operator!=(InternalState s)
{
	// compare state and correpsonding observed state
	return (state != s.state || extStateNum != s.extStateNum);
}

bool InternalState::operator<(InternalState& s)
{
	// compare state and correpsonding observed state
	return (distance < s.distance);
}

/*
	Function: genSuccessors()
	Desc.	: Generate all the possible successors base on current state and primitive action
	Para.	: actions, the set of primitive actions
			  intObjects, the set of internal objects
	Return	: return all the possible successor of the current state
	Note	: Here only correct parameters are consider to each action.
			  Assume the first element in intObjects is objA
*/
list<InternalState> InternalState::genSuccessors(vector<Action> actions, vector<Object> intObjects) 
{	
	size_t i, j;
	const size_t ACTION_SIZE = actions.size();
	const size_t OBJECTS_SIZE = intObjects.size();

	list<InternalState> successors, s;
	
	// for each action, check possible
	for (i=0; i<ACTION_SIZE; ++i) 
	{
		switch (i) {
			case 0:		// MOVE
				// can MOVE to any objects
				for (j=0;j<OBJECTS_SIZE;++j)
				{
					s = genASuccessor(actions[i], intObjects[j].name, "");
					successors.insert(successors.end(), s.begin(), s.end());
				}
				break;
			case 1:		// GRAB
			case 2:		// DROP
				for (j=0; j<OBJECTS_SIZE; ++j)
				{
					// can GRAB/DROP objA, Toy, futon1 and futon2
					if (intObjects[j].name == "ObjA" || intObjects[j].name == "ObjB" || intObjects[j].name == "Toy" ||
						intObjects[j].name == "Futon1" || intObjects[j].name == "Futon2")
					{
						s = genASuccessor(actions[i], intObjects[j].name, "");
						successors.insert(successors.end(), s.begin(), s.end());
					}
				}
				break;
			case 3:		// PUSH 1st 2nd, push 2nd toward 1st
				// can PUSH the 1st object(except Funton1&2) to anywhere
				if (intObjects[0].name != "Futon1" && intObjects[0].name != "Futon2")
				{
					for (j=1;j<OBJECTS_SIZE;++j)
					{
						//s = genASuccessor(actions[i], intObjects[0].name, intObjects[j].name);
						s = genASuccessor(actions[i], intObjects[j].name, intObjects[0].name);
						successors.insert(successors.end(), s.begin(), s.end());
					}
				}

				// for the 2nd object, check whether if it is objA, Toy
				if (intObjects[1].name == "ObjA" || intObjects[1].name == "Toy")
				{
					// PUSH toward 1st object
					/*s = genASuccessor(actions[i], intObjects[1].name, intObjects[0].name);*/
					s = genASuccessor(actions[i], intObjects[0].name, intObjects[1].name);
					successors.insert(successors.end(), s.begin(), s.end());

					// PUSH toward 3rd object
					/*s = genASuccessor(actions[i], intObjects[1].name, intObjects[2].name);*/
					s = genASuccessor(actions[i], intObjects[2].name, intObjects[1].name);
					successors.insert(successors.end(), s.begin(), s.end());
				}
				break;
		}
	}
	// one addtional state, without taking any action, move to next observed state
	successors.push_back(InternalState(-1, state, extStateNum+1));

	return successors;
}

/*
	Function: genASuccessor()
	Desc.	: generate the successors base on current state and one action
	Para.	: action, the action to take to generate its successors
			  p1, actual parameter 1
			  p2, actual parameter 2
	Return	: None.
	Note	: This method generates two successors, one corresponds to the next observed state, 
			  the other corresponds to the current observed state
*/
list<InternalState> InternalState::genASuccessor(Action action, string p1, string p2) 
{	
	//Action a;		// action that initialized with concrete object
	State nextState;
	
	list<InternalState> successors;

	//base on parameter, initialize the action's precondition/postcondition
	action.parameterize(p1,p2);

	//check whether action's precondition is satisfied
	if (action.IsSatisfied(state)){
		//execute the action if satisfied
		nextState = action.Execute(state);

		// new internal state still correspond to current observed state
		successors.push_back(InternalState(action.num, nextState, extStateNum));

		// new internal state correspond to next observed state
		successors.push_back(InternalState(action.num, nextState, extStateNum+1));
	}

	return successors;
}

/*
	Function: toString()
	Desc.	: Returns a String that represents this instance.
	Para.	: None
	Return	: string
*/
string InternalState::toString() const
{
	string s;

	s = convertToString(action) + " " + convertToString(distance) + " " + convertToString(extStateNum) +  "\n";
	s+=state.toString();
	
	return s;
}