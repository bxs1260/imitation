#ifndef ACTION_H
#define ACTION_H

#include <string>

#include "State.h"
#include "Utility.h"

using namespace std;

class Action
{
	void Action::parameterize(State& state, string p1, string p2);

	vector<string> paras;	// parameter for execution

public:
	int	num;
	string name;
	State preConds;			// precondition, vary
	State postConds;		// postcondition, vary
	double cost;			// cost of the action

	Action(void);
	Action(int numVal, string nameVal, State preCondsVal, State postCondsVal, double costVal);

	~Action(void);

	// clean this instance.
	void clear();
	
	// Given a state, execute this instance and return next state
	State Execute(const State& currState);
	
	// check whether this instance's precondition is satisfied by the current state
	bool IsSatisfied(State currState) ;

	// Fill this instance with real parameter
	void parameterize(string p1, string p2);

	// Returns a String that represents this instance
	string toString() const;
};
#endif
