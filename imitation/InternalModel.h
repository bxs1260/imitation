#ifndef INTERNALMODEL_H
#define INTERNALMODEL_H

#include <vector>

#include "Object.h"
#include "InternalState.h"

using namespace std;
class Node
{
	public:
		int level;			// level in the A* tree
		InternalState state;
		double g;			// the cost of getting from the initial node to this instance
		double h;			// the estimate cost of getting from this instance to the goal node.
		double f;			// total cost, g+h

		Node(void);
		Node(InternalState stateVal, int levelVal, double gVal, double hVal);

		~Node(void);

		string toString() const;
		bool operator==(const Node &node);
		bool operator!=(const Node &node);
		bool operator<(const Node node);
};

class InternalModel
{
public:
	vector<Node> policy;		// internal policy
	vector<Node> siblings;		// internal A* tree
	vector<Object> objects;		// internal objects
	double reward;

	InternalModel(void);
	InternalModel(vector<Object> objectsVal, vector<Node> policyVal, vector<Node> siblingsVal, double rewardVal=0);

	~InternalModel(void);

	string toString() const;
};
#endif
