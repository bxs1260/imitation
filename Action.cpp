#include "Action.h"

Action::Action(void) {}

Action::Action(int numVal, string nameVal, State preCondsVal, State postCondsVal, double costVal)
: num(numVal), name(nameVal), preConds(preCondsVal), postConds(postCondsVal), cost(costVal) {}

Action::~Action(void) {}

/*
	Function: parameterize
	Desc.	: Fill this instance with real parameter
	Para.	: p1, parameter 1
			  p2, parameter 2
	Return	: Action
*/
void Action::parameterize(string p1, string p2)
{
	// parameterize pre-condition
	parameterize(preConds, p1, p2);
	
	// parameterize pos-condition
	parameterize(postConds, p1, p2);
	
	// parameters
	paras.clear();
	paras.push_back(p1);
	paras.push_back(p2);
}

/*
	Function: parameterize
	Desc.	: Fill preConds/postConds with real parameters
	Para.	: state, original representation
			  p1, parameter 1
			  p2, parameter 2
	Return	: State
*/
void Action::parameterize(State& state, string p1, string p2)
{
	int i;
	
	for (i=0; i<state.size(); ++i)
	{
		if (state[i].objA=="PARA_1") {
			state[i].objA=p1;
		} else if (state[i].objA=="PARA_2") {
			state[i].objA=p2;
		}
			
		if (state[i].objB=="PARA_1") {
			state[i].objB=p1;
		} else if (state[i].objB=="PARA_2") {
			state[i].objB=p2;
		}
	}
}

/*
	Function: Execute()
	Desc.	: Given a state, execute this instance and return next state
	Para.	: currState, current state
	Return	: next state after executing this instance
	Note	: need call ParameterizeAction and IsSatisfied() first.
*/
State Action::Execute(const State& currState)
{
	int i;
	State nextState;

	vector<string>::iterator p;
	vector<Relation>::iterator q;

	//update next state when action's preconditon is satisfied
	nextState = currState;

	// remove precond from next state
	nextState.remove(preConds);
	
	// check current action's number
	switch (num) {
		case 0:		// MOVE
		case 3:		// PUSH
			// for MOVE/PUSH action, change other "NEXT" to "AWAY" which are not included in preConds
			for (i=0;i<nextState.size();++i)
				if (nextState[i].relation=="NEXT")
					nextState[i].relation="AWAY";	// hardcode here, change later

			// if other objects are next to the target object, update the relation between those objects and imitator
			if (find(nextState.nextTo.begin(), nextState.nextTo.end(), paras[0])!=nextState.nextTo.end())
			{
				for (i=0; i<nextState.nextTo.size(); ++i)
				{
					q = find(nextState.begin(), nextState.end(), Relation("AWAY", "Imitator", nextState.nextTo[i]));
					if (q!=nextState.end())
						q->relation = "NEXT";
				}
			}

			// for PUSH action
			if (num==3)
			{
				// push away
				p= find(nextState.nextTo.begin(), nextState.nextTo.end(), paras[1]);
				if (p!=nextState.nextTo.end())
					nextState.nextTo.erase(p);
				
				if (nextState.nextTo.size()<2)
					nextState.nextTo.clear();

				// push toward
				nextState.nextTo.push_back(paras[1]);
				p= find(nextState.nextTo.begin(), nextState.nextTo.end(), paras[0]);
				if (p!=nextState.nextTo.end())
					nextState.nextTo.push_back(paras[0]);
			}
			break;
		case 1:		// GRAB
			// check whether the object grabed is next to other object before
			p= find(nextState.nextTo.begin(), nextState.nextTo.end(), paras[0]);
			if (p!=nextState.nextTo.end())
				nextState.nextTo.erase(p);
			
			if (nextState.nextTo.size()<2)
				nextState.nextTo.clear();

			break;
		case 2:		// DROP
			Relation r("NEXT", "Imitator", "?");
			vector<Relation>::iterator p = find(nextState.begin(), nextState.end(), r);
			if (p!=nextState.end())
			{
				if (nextState.nextTo.size() == 0)
					nextState.nextTo.push_back(p->objB);

				// If there is something next to the imitator, add this object to nextTo
				nextState.nextTo.push_back(paras[0]);
			}

			// object disappear when drop into trashcan
			r = Relation("NEXT", "Imitator", "Trashcan");
			
			// check if there is trashcan next to it, if not add "NEXT imitator objA", otherwise the objA will be disappear	
			p=find(nextState.begin(), nextState.end(), r);
			if (p==nextState.end())
				// there is no trashcan next to it, add additional relation to the next state.
				nextState.add(Relation("NEXT","Imitator",preConds[0].objB));
			break;
	}

	// add postcond to next state
	nextState.add(postConds);

	return nextState;
}

/*
	Function: IsSatisfied()
	Desc.	: check whether this instance's precondition is satisfied by the current state
	Para.	: currState, current state
	Return	: bool, true when satisfied, otherwise false.
*/
bool Action::IsSatisfied(State currState)
{
	return (currState >= preConds);
}

/*
	Function: clear()
	Desc.	: clean this instance.
	Para.	: None
	Return	: None
*/
void Action::clear()
{
	num=-1;
	name.clear();
	preConds.clear();
	postConds.clear();
	cost = 0;
}

/*
	Function: toString()
	Desc.	: Returns a String that represents this instance
	Para.	: None
	Return	: string
*/
string Action::toString() const
{
	string s;
	
	s = convertToString(num) + " " + name + " " + convertToString(cost) + "\n";
	
	s+=preConds.toString();
	s+=postConds.toString();

	return s;
}