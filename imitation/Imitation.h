#ifndef IMITATION_H
#define IMITATION_H

#include <map>
#include <vector>
#include <string>
#include <fstream>
//#include <conio.h>
#include <functional>
#include <algorithm>
#include <cassert>
#include <list>

#include "InternalModel.h"
#include "InternalState.h"
#include "FeedForward.h"

#include "Object.h"
#include "Relation.h"
#include "State.h"
#include "Action.h"
#include "ObservedModel.h"
#include "Utility.h"
#include "Random.h"
#include "tree.h"

const int INIT_STD_DEV = 30;		// initial standard deviation
const int COST_ATTRIBUTE = 30;

const int NUM_OF_INPUT = 32;		// number of input units

const int LEARNING_ROUND = 100;		//100			// the number of rounds occur for a set of tasks
const int BATCH_UPDATE_ITERATION = 500;	// how many iterations happened in each batch update

const double MARGIN_PER = 0.3;			//0.6 for fix amount, 0.3 for percentage;
const double MARGIN = 0.6;
const double DISCOUNT_FACTOR = 0.995;

// task definition
enum {TRASH_CLEANING, TOY_COLLECTION, FUTON_MATCH_1, FUTON_MATCH_2};

const string DEMO = "Demo";
const string NNFILE = "nn.txt";	// neural network's file

// policy & sibling type
typedef struct pair<vector<Node>, vector<Node> > psType;
typedef tree_node_<Node> treeNode;

using namespace std;

class Imitation
{
private:
	/****************************************** internal variables ****************************************/
	// the primitive actions
	vector<Action> actions;

	// standard deviation
	double stdDeviation;

	// sample data, use to initialize neural network
	vector<vector<double> > samples;
	
	// objects in the internal model
	vector<Object> intObjects;
	vector<vector<Object> > imitObjects;
	
	// successors generated from current state	
	vector<InternalState> successors;

	// numeric representation for state pairs
	vector<vector<double> > inputs;

	double currReward;
	// output from exploration, also contains count and rewardDiff for the tasks in which it appeared in their A* tree
	//	- current output
	//	- the sum of (delta* rewardDiff) across all the tasks
	vector<vector<double> > exploredOutputs;
	// the sum of reward difference for each state pair, separate from exploredOutputs make it easy to get the maximum one
	vector<double> rewardDiffs;

	// expected output for each state pairs in the inputs
	vector<double> expectedOutputs;

	// mapping between the observed state and internal state
	map<string, string> mMap;

	// numeric representation of observed state and internal state
	map<string, double> extNumMap, intNumMap;

	/*********************************** variable and method for A* algorithm *****************************/
	// open list stores the nodes that have not been expanded, closed list stored the nodes that have been expaned.
	list<treeNode *> openList, closedList;
	
	// current A* tree, only for the current A* search
	tree<Node> currAStarTree, newAStarTree;
	
	psType currPolicySibling;

	/******************************* variables represent external objects **********************************/	
	// use for exploration
	Random r;

	// FeedForward neural network
	FeedForward nn;
	int numOfHiddenUnits; 

	// observed mode
	ObservedModel extModel;

	// current observed model, only contain one demonstration
	vector<Object> currObservedObjects;
	vector<State> currObservedStates;
	
	/********************************************** Method *************************************************/

	// add a state into a specified list
	void addToList(list<treeNode*> &target, treeNode *node);
	void removeFromList(const tree<Node>& aStarTree, treeNode *node);

	// calculate distance between the observed an mapped state
	double calcDistance(State extState, State intState, int modelState);	// symbol representation input
	double calcDistance(vector<double> input);								// numeric representation input

	// convert observed state and internal state into a numeric representation which will be provided to RBF-NN as input
	vector<double> convert(State extState, State intState);
	vector<double> convert(State state, bool internal);

	// base on demonstration generate a set of sample which is used to initialize the neural network, for multiple single-step demonstrations
	void generateSamples();
	
	// load primitive actions from file
	void loadAction(const string fileName);

	// load mapping between observed state and internal state
	void loadMapping(const string fileName);

	// load the numeric representation of observed/internal model
	map<string, double> loadNumMapping(const string fileName);

	// load previous learned demonstrations
	void load();
	
	// load a previous learned demonstration
	void loadLearnedDemo(const string fileName);

	// load objects for imitation
	void loadImitObjects(const string fileName);
	
	// save demonstrations learned
	void save();				// number and file name
	void saveLearnedDemos();	// demonstrations just learned

	// set current observed model, always call no matter single or multiple demonstration(s) 
	void setCurrentObservedModel(const vector<Object> &observedObjects, const vector<State> &observedStates);
	void setCurrentObservedModel(const vector<Object> &observedObjects, const vector<State> &observedStates, const vector<Object> internalObjects);

	/********************************** Method related to A* algorithm ***********************************/
	void printTree(fstream &fout,  const tree<Node>& aStarTree, bool standardOutput=false);
	// using A* algorithm to find a policy
	psType AStarSearch(int modelState, tree<Node>& aStarTree);

	// get the policy
	psType getPolicy(const tree<Node>& tree, bool lBackpropagate=true);
	
	// backpropagate heuristic cost
	void backpropagateHeuristicCost(tree<Node>&);
	void backpropagateHeuristicCost(vector<Node>&);
	
	/***************************************** Policy Exploration *******************************************/
	// do a cost exploration on each successor and reorder them
	void chooseASuccessor(list<InternalState> &successors);
	
	// calculate distance from a given policy (mapping the policy back to the distance representation)
	void batchUpdate(double totRewardDiff);

	// check whether the policy in this internal model is satisfied
	void generateDistance(InternalModel &intModel, double rewardDiff=0);
	bool policyIsSatisfied(ObservedModel &demo, InternalModel &imit);
	
	// add the input/output into training set
	void addToTrainingSet(const InternalState& state, double expectedOutput, double rewardDiff = 0, double delta = 0);

	// print A* tree
	void printNodes(fstream &fout, const vector<Node>& nodes, bool standardOutput=false);

	/************************************** Miscellaneous Method ******************************************/
	// load input/expected output from a file
	void loadData(string fileName);
	// save input/expected output into a file
	void saveData(string fileName);

	// find the position when found, special purpose
	int search(vector<vector<double> > &all , vector<double> single);

	// new and learned demonstration
	vector<ObservedModel> newDemos, learnedDemos;

	// new and learned internal model, one demonstration may have multiple imitation for different case
	vector<vector<InternalModel> > newImits, learnedImits;

	// the index of current imitation for each demo
	vector<int> idxOfCurrImits;
	
	// recalculate the distance in the internal model
	void recalcDistance(InternalModel &intModel);
	// recalculate the distance for a single node
	void recalcDistance(Node &node, double gOfParent);
	
	// calculate action's cost given its num
	double calcActionCost(int num);

	vector<vector<string> > imitationEnv;
	fstream fout_update, fout_policy, fout_err, fout_rew, fout_oldRew;

	// calculate reward for different task
	double calcReward(vector<Node> &policy, int numOfDemo = -1);
	
	double calcReward(State state);
	double calcRewardForCleaning(State state);
	double calcRewardForCollection(State state);
	double calcRewardForDoubleDrop(State state);

	void changeImitationEnvironment(int numOfDemo, int idxOfAttr = -1);
	void clearTrainingSet();

	// check whether the given state is goal state
	bool isGoalState(const InternalState& intState);
	double calcHeuristicCost(const InternalState& intState);

	// another version of calculate the distance between observed state and internal state, comparing with neural network
	double handCode(State& extState, State& intState);
	vector<vector<string> > stateToString(State state, bool internal=true);

	double similar(const vector<string>& extState, const vector<string>& intState);
	double minMax(tree<Node>& aStarTree, tree<Node>::iterator_base&);
	
	void testAction(State& s, int iAction, string p1, string p2="");

	
	bool DEBUG_MODE;		// when DEBUG_MODE is true, output experiment's detail 


	/**************************    Utilities      ***********************************/
	void test_1(int modelState, fstream &fout);
	void test_N(int modelState, fstream &fout);
	void test_new(int modelState, fstream &fout);

	double simpleDistance(const State& extState, const State& intState);
public:
	Imitation(int numOfHidden = 10, bool debugModel = false);
	~Imitation(void);

	// internal model's state
	enum {
		EXPLORATION,	// generate a distance between observed state and internal state based on gaussian distribution
		EXPLOITATION,	// calculate distance based on current network configuration
		HANDCODE,		// calculate distance using hand-code function
		HANDCODE_EW		// hand-coded with equal weight
	};


	// update the network to produce optimal policy for given demonstration
	void training();
	
	// given a observed model, try to imitate
	void learning(string fileName);
	
	// given a demonstration, testing the imitator's capability
	void testing(int modelState = EXPLOITATION);

	// load observed model from a file
	void loadNewDemos(string fileName);

	// generate internalModel for each task from NN configurate
	void generatePSFromNN();
};

	// Nonmember functions

	// map object, relation, state into internal representation
	Object mapping(const Object& o, map<string, string>& m);
	Relation mapping(const Relation& r, map<string, string>& m);
	State mapping(State s, map<string,string>& m);
	
	// create object from input
	Object readObject(fstream &fin);
	Relation readRelation(fstream &fin);
	State readState(fstream &fin);
	Action readAction(fstream &fin);
	InternalState readInternalState(fstream &fin);
	ObservedModel readObservedModel(fstream &fin);
	InternalModel readInternalModel(fstream &fin);

#endif
