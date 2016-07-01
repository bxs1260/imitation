#include "Relation.h"

Relation::Relation(string relationVal, string objAVal, string objBVal)
: objA(objAVal), objB(objBVal), relation(relationVal){}

Relation::~Relation(void) {}

/*
	Function: == operator
	Desc.	: compares this instance with a specified relation
	Para.	: r, the relations that are going to be compared.
	Return	: bool, true when they are equal, otherwise return false
*/
bool Relation::operator==(const Relation& r) const
{
	return ((r.objA == "?" || objA==r.objA) && (r.objB == "?" || objB==r.objB) && (r.relation == "?" || relation==r.relation));
}

/*
	Function: toString()
	Desc.	: Returns a String that represents this instance.
	Para.	: None
	Return	: string
*/
string Relation::toString() const
{
	return relation + " " + objA + " " + objB + "\n";
}