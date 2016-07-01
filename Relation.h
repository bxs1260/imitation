#ifndef RELATION_H
#define RELATION_H

#include <string>

using namespace std;
class Relation
{
public:
	/* attribute */ 
	string objA;
	string objB;
	string relation;
	
	/* constructor */
	Relation(string relationVal="", string objAVal="", string objBVal="");
	~Relation(void);

	/* Method */
	bool operator==(const Relation& r) const;
	
	// returns a String that represents this instance.
	string toString() const;
};
#endif
