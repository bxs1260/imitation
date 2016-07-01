#ifndef INTERNALSTATE_H
#define INTERNALSTATE_H

#include "State.h"
#include "Action.h"
#include "Utility.h"
#include "Object.h"

#include <list>
using namespace std;

class InternalState
{
	// generate the successors base on current state and one action
	list<InternalState> genASuccessor(Action action, string p1, string p2);

public:
	int action;			// action taken at previous states that end up with this instance
	State state;		// internal state
	int extStateNum;	// corresponding observed state
	double distance;	// distance between observed state and internal state

	InternalState(void);
	InternalState(int actionVal, State stateVal, int extStateNumVal, double distanceVal=0);

	~InternalState(void);

	// clear this instance
	void clear();

	// generate all the possible successors base on current state and primitive action
	list<InternalState> genSuccessors(vector<Action> actions, vector<Object> intObjects);

	// check whether the given state is equal to this instance, state and extStateNum comparison
	bool operator==(InternalState s);
	bool operator!=(InternalState s);
	bool operator<(InternalState& s);
	
	// Returns a String that represents this instance
	string toString() const;
};
#endif
