#ifndef STATE_H
#define STATE_H

#include <vector>
#include <iterator>
#include <algorithm>

#include "Relation.h"
#include "Utility.h"

using namespace std;

class Action;

class State
{
private:
	vector<Relation> state;

protected:
	vector<string> nextTo;
	friend class Action;

public:
	State(void);
	~State(void);

	// add a relation to this instance
	void add(Relation added);

	// add all the relations in the specified state to this instance
	void add(State added);

	// clear this instance
	void clear();

	// check whether a relation exists in a relation container
	vector<Relation>::const_iterator findPattern(const Relation& pattern) const;

	// compares this instance with a specified state
	bool operator==(State s);
	bool operator!=(State s);
	bool operator>=(State s);

	// overload the [] subscript
	Relation &operator[](int i);

	// remove all the relations of the specified state from this instance
	void remove(State removed);

	//// return the size of the state
	int size() const;

	// returns a String that represents this instance.
	string toString() const;
	string nextToObjects() const;

	vector<Relation>::iterator begin();
	vector<Relation>::iterator end();

	// update nextTo relation between objects
	void updateNextTo();
};

#endif
