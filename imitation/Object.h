#ifndef OBJECT_H
#define OBJECT_H

#include <string>
#include <functional>

using namespace std;
class Object
{
public:
	string name;
	string color;
	string texture;

	Object(string nameVal="", string colorVal="", string textureVal="");

	~Object(void);

	// operator
	bool operator==(const Object& o) const;

	string toString() const;
};
#endif
