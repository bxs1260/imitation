#ifndef UTILITY_H
#define UTILITY_H

#include <sstream>
#include <vector>
#include <functional>
#include <cstdlib>

using namespace std;

/*
	Function: convertToString
	Desc.	: a generic conversion template that converts from arbitrary types to string
	Para.	: t, source representation need to be converted
	Return	: target representation specified by out_type
*/
template <class in_value>
string convertToString(const in_value & t)
{
	string result;
	stringstream stream;

	stream << t; // insert value to stream
	stream >> result; // write value to result

	return result;
};

/*
	Function: sign()
	Desc	: 
	Para	:
	Return	: -1 when x>0, 0 when x=0, -1 when x<0

*/
template <class in_type>
int sign(in_type x)
{
	if (x>0)
		return 1;
	else if (x==0)
		return 0;
	else
		return -1;
};

/*
	Function: shuffle
	Desc.	: Rearrange two related sets (usually inputs/outputs) to produce a random order
	Para.	: in_value1, a set of elements need to be rearrange, usually is inputs set
			  in_value2, a set of elements need to be rearrange, usually is outputs set
	Return	: None
	Note	: These two sets must have same length
*/
template <class in_type1, class in_type2>
void shuffle(vector<in_type1> &in_value1, vector<in_type2> &in_value2)
{
	int i, r;
	in_type1 t1;
	in_type2 t2;

    // Shuffle elements by randomly exchanging each with one other.
    for (i=0; i<in_value1.size(); ++i) 
	{
        r = rand() % in_value1.size();  // generate a random position
		t1 = in_value1[i];
		in_value1[i]=in_value1[r];
		in_value1[r]=t1;

		// swape element in the second parameter
		t2 = in_value2[i];
		in_value2[i]=in_value2[r];
		in_value2[r]=t2;
    }
};

template <class T> 
class sameName : public unary_function <T, bool>
{
	string s;
public:
	explicit sameName(const string& val) : s(val) {}
	bool operator() (const T& o) const { return o.name == s; }
};

#endif
