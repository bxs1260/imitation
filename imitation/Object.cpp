#include "Object.h"

Object::Object(string nameVal, string colorVal, string textureVal) : 
	name(nameVal), color(colorVal), texture(textureVal) {}

Object::~Object(void){}

string Object::toString() const
{
	return name + " " + color + " " + texture + "\n";
}

bool Object::operator==(const Object& o) const
{
	return ((o.name == "?" || name==o.name) && (o.color == "?" || color==o.color) && (o.texture == "?" || texture ==o.texture));
}