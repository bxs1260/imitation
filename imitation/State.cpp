#include "State.h"

State::State(void) {}
State::~State(void) {}

/*
	Function: operator >=
	Desc	: check whether this instance is large or equal to the given state s
	Para.	: s, the state that are going to be compared
	Return	: return true if each element in state s can be found in this instance
	Note	: These two states can be equal.
			  This method can be used to check whether one action's precondition is satisifed in current state
*/
bool State::operator>=(State s)
{
	int i;
	vector<Relation>::const_iterator p;

	// check each element in state s
	for (i=0; i<s.size(); ++i)
	{
		p = find(state.begin(), state.end(), s[i]);
		if (p == state.end())
			// doen't exist
			return false;
	}
	return true;
}

/*	
	Function: operator==
	Desc.	: compares this instance with a specified state
	Para.	: s, the state that are going to be compared
	Return	: return true if equal, otherwise return false
*/
bool State::operator==(State s)
{
	if (state.size()!=s.size()) 
		return false;
	
	return (*this >= s);
}

/*	
	Function: operator!=
	Desc.	: compares this instance with a specified state
	Para.	: s, the state that are going to be compared
	Return	: return true if not equal, otherwise return false
*/
bool State::operator!=(State s)
{
	return !(*this==s);
}

/*
	Function: size()
	Desc.	: return the size of this instance
	Para.	: None
	Return	: size of this instance
*/
int State::size() const
{
	return state.size();
}

/*
	Function: remove()
	Desc.	: Remove all the relations of the specified state from this instance.
	Para.	: removed, the relations that are going to be removed from this instance
	Return	: None
	Note	: This method is used to remove precondition from current state after execute an action
*/
void State::remove(State removed) 
{
	int i;
	vector<Relation>::iterator p;

	for (i=0; i<removed.size(); ++i)
	{
		// find the position of the relation which is going to be removed
		p=find(state.begin(), state.end(), removed[i]);
		
		// remove from this instance
		if (p!=state.end())
			state.erase(p);
	}
}

/*
	Function: add()
	Desc.	: Add all the relations in the specified state into this instance
	Para.	: added, the state that are going to be added.
	Return	: None
	Note	: This method is used to add postcondition to current state after execute an action
*/
void State::add(State added) 
{
	state.insert(state.end(), added.begin(), added.end());
}

/*
	Function: add()
	Desc.	: Add one relation to this instance
	Para.	: added, the relation that are going to be added.
	Return	: None
*/
void State::add(Relation added) 
{
	state.push_back(added);
}

/*
	Function: operator[]
	Desc.	: overload the [] subscript
	Para.	: i, subscript
	Return	: Relation, one relation in this instance specified by the subscript
*/
Relation &State::operator[](int i)
{
	return state[i];
}

/*
	Function: clear()
	Desc.	: Clear this instance
	Para.	: None
	Return	: None
*/
void State::clear()
{
	state.clear();
}

/*
	Function: findPattern()
	Desc.	: check whether the given relation exists in this instance or not
	Para.	: relation, the relation we are looking for in this instance
			: ignoreRelation, default is true, don't compare the relation component. 
	Return	: if there is a match, return its position; otherwise return -1
	Note	: the main purpose of this function is check whether a relation exists in a pattern
*/
vector<Relation>::const_iterator State::findPattern(const Relation& pattern) const
{	
	return find(state.begin(), state.end(), pattern);
}
/*
	Function: toString()
	Desc.	: Returns a String that represents this instance.
	Para.	: None
	Return	: string
*/
string State::toString() const
{
	string s;
	const size_t N = state.size();
	
	// number of relations in this instance
	s = convertToString(N) + "\n";
	for (size_t i=0; i<N; ++i)
		s+=state[i].toString();

	return s;
}

string State::nextToObjects() const
{
	string s = "";
	for (size_t i=0; i<nextTo.size(); ++i)
		s += nextTo[i] + " ";

	return s;
}
vector<Relation>::iterator State::begin()
{
	return state.begin();
}

vector<Relation>::iterator State::end()
{
	return state.end();
}

/*
	Function: updateNextTo()
	Desc.	: see which objects in the current state are next to each other
	Para.	: None
	Return	: None
*/
void State::updateNextTo()
{
	int i;
	
	nextTo.clear();
	for (i=0; i<state.size(); ++i)
		if (state[i].relation == "NEXT")
			nextTo.push_back(state[i].objB);
	
	// if only next to one object, ignore
	if (nextTo.size()<2)
		nextTo.clear();
}