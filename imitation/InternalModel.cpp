#include "InternalModel.h"

Node::Node(void) {}
Node::~Node(void) {}

Node::Node(InternalState stateVal, int levelVal, double gVal, double hVal)
: state(stateVal), level(levelVal), g(gVal), h(hVal)
{
	f = g + h;
}

string Node::toString() const
{ 
	// level, the estimate cost h and the state detail
	return convertToString(level) + " " + convertToString(h) + "\n" + state.toString(); 
}

bool Node::operator==(const Node &node)
{
	return (state==node.state);
}

bool Node::operator!=(const Node &node)
{
	return (state!=node.state);
}
bool Node::operator<(const Node node)
{
	return (level<node.level || (level==node.level && f<node.f));
}

/* Internal Model */
InternalModel::InternalModel(void) {}
InternalModel::~InternalModel(void) {}

InternalModel::InternalModel(vector<Object> objectsVal, vector<Node> policyVal, vector<Node> siblingsVal, double rewardVal)
: policy(policyVal), siblings(siblingsVal), objects(objectsVal), reward(rewardVal) {}

string InternalModel::toString() const
{
	size_t i, N;
	string s;
	vector<Node>::const_iterator iter;

	// its reward and number of nodes in the A* tree
	s = convertToString(reward) + " " + convertToString(policy.size()+ siblings.size()) + "\n";

	// number of internal objects
	N = objects.size();
	s += convertToString(N) + "\n";
	// objects detail
	for (i=0; i<N; ++i)
		s += objects[i].toString();
	s+= "\n";

	// the policy, chosen flag is 1
	for (iter = policy.begin(); iter != policy.end(); ++iter)
		s += convertToString(1) + " " + iter->toString();
	s+= "\n";

	// A* tree, chosen flag is 0
	for (iter = siblings.begin(); iter != siblings.end(); ++iter)
		s += convertToString(0) + " " + iter->toString(); 
	s+= "\n";

	return s;
}
