#include "Imitation.h"

Imitation::Imitation(int numOfHidden, bool debugMode)
{
	numOfHiddenUnits = numOfHidden;
	DEBUG_MODE = debugMode;

	// load primitive action
	loadAction("actions.txt");

	// load mapping between observed model and internal model	
	loadMapping("mapping.txt");

	// load numeric representation of external state
	extNumMap = loadNumMapping("externalNumRep.txt");

	// load the numeric representation of internal model from file
	intNumMap = loadNumMapping("internalNumRep.txt");

	// load learned demonstration
	load();

	int i, j, num, idxOfDemo, numOfDemos;
	string attr;
	fstream fin;

	vector<string> attrs;

	fin.open("imitationEnv.txt", ios::in);
	fin >> numOfDemos;
	for (i=0; i<numOfDemos; ++i)
	{	
		// ignore demo's num
		fin >> idxOfDemo;

		fin >> num;

		attrs.clear();
		for (j=0; j<num; ++j)
		{
			fin >> attr;
			attrs.push_back(attr);
		}
			imitationEnv.push_back(attrs);
	}
	fin.close();

	fout_update.open("p_update.txt", ios::out);
	fout_policy.open("p_policy.txt", ios::out);
	fout_err.open("p_err.txt", ios::out);
	
	fout_rew.open("o_reward.txt", ios::out);
	fout_oldRew.open ("o_old_rew.txt", ios::out);
}

Imitation::~Imitation(void) 
{
	fout_update.close();
	fout_policy.close();
	fout_err.close();
	
	fout_rew.close();
	fout_oldRew.close();
}

/*
	Function: calcDistance()
	Desc.	: Using NN to calculate the distance between the observed state and internal state
	Para.	: Observed and internal state and current model
				- extState, observed state
				- intState, internal state				
				- model: EXPLOITATION/EXPLORATION
					* EXPLOITATION: call calcOutput() and return the result.
					* EXPLORATION: geneerate new cost based on its mean (the result from calcRBF()) and standard deviation.
	Return	: The distance between observed state and internal state
*/
double Imitation::calcDistance(State extState, State intState, int modelState)
{
	double mean, rnd;
	vector<double> input;

	// convert state pair into numeric representation
	input = convert(extState, intState);

	// calculate distance
	mean = calcDistance(input);

	// EXPLOITATION phase, directly return the result from NN
	if (modelState == EXPLOITATION)
		return mean;

	// EXPLORATION phase, generate new cost base on current mean and standard deviation
	while (true)
	{
		rnd = r.nextGaussian(mean, stdDeviation);
		if (rnd>=0)
			return rnd;
	}
}

/*
	Function: calcDistance()
	Desc.	: Using NN to calculate the distance between the observed state and internal state
	Para.	: input, numeric representation for Observed and internal state
	Return	: The distance between observed state and internal state
*/
double Imitation::calcDistance(vector<double> input)
{
	double output;

	output = nn.calcOutput(input);
	// set it zero when it is negative
	if (output < 0)
		output = 0;
	
	return output;
}

/*
	Function: loadAction()
	Desc.	: load primitive action from file.
	Para.	: fileName, file name
	Return	: None
*/
void Imitation::loadAction(const string fileName)
{
	int i, num;	
	fstream fin;

	// open action file
	fin.open(fileName.c_str(), ios::in);

	// read num of actions
	fin >> num;
	for (i=0; i<num; ++i)	
		actions.push_back(readAction(fin));
	
	fin.close();
}

/* 
	Function: convert()
	Desc.	: convert observed/internal state representation from symbol to numberic, which are used as input to neural network
	Para.	: extState, observed state
			  intState, internal state
	Return	: numeric representation of the observed state and internal state
*/
vector<double> Imitation::convert(State extState, State intState)
{
	vector<double> v1, v2;

	// convert observed state
	v1 = convert(extState, false);

	// convert internal state
	v2 = convert(intState, true);

	// append internal representation at the end
	v1.insert(v1.end(), v2.begin(), v2.end());

	return v1;
}

/*
	Function: convert()
	Desc	: Given a state and return its numeric representation which encoding both relationship and objects' attributes
	Para	: state, a state representation
			  internal, whether the state is internal represetnation or not
	Return	: the numeric representation
*/
vector<double> Imitation::convert(State state, bool internal)
{
	size_t i;
	vector<Relation>::const_iterator iter;
	vector<double> result;
	
	string actor;
	vector<Object> o;				// objects
	map<string,double> m;			// map between string representation and numeric representation
	
	if (internal)
	{
		//actor = mapto<string>(DEMO, mMap);
		actor = mMap[DEMO];			// TEST ON 12/22/05
		o = intObjects;				// internal object representation
		m = intNumMap;
	}
	else
	{
		actor = DEMO;
		o = currObservedObjects;	// observed object representation
		m = extNumMap;
	}
	
	// assume each demonstration has a fully object definition, even if in some case just use part of it.
	// Object name isn't encoded in the input, but its attributes, color and texture
	for (i=0; i<o.size(); ++i)
	{
		// check if the same entry with "Demo objA" exists in the observed/internal state
		iter = state.findPattern(Relation("?", actor, o[i].name));
		if (iter == state.end())
			// not exists in the state, insert four 0 (relation, Demo and two attributes)
			result.insert(result.end(),4, 0);
		else
		{
			// relation
			result.push_back(m[iter->relation]);
			// demonstrator or imitator
			result.push_back(m[iter->objA]);

			// object attributes
			result.push_back(m[o[i].color]);
			result.push_back(m[o[i].texture]);
		}
	}

	 // There is special case for gripper, check if an object on the gripper
	iter = state.findPattern(Relation("ON", "Gripper", "?"));
	result.push_back(m["ON"]);
	result.push_back(m["Gripper"]);
	
	if (iter->objB == "NULL")
		// nothing on gripper, two 0s for attribute
		result.insert(result.end(),2, 0);		// NULL or 0??
	else
	{
		// find the object attributes
		vector<Object>::const_iterator objIter = find_if(o.begin(), o.end(), sameName<Object>(iter->objB));
		result.push_back(m[objIter->color]);
		result.push_back(m[objIter->texture]);
	}

	return result;
}
	
int Imitation::search(vector<vector<double> > &all , vector<double> single)
{
	size_t i;
	for (i=0; i<all.size(); ++i)
		if (all[i] == single)
			return i;
	return -1;
}
/*
	Function: loadMapping()
	Desc.	: load the mapping between observed model and internal model
	Para.	: fileName, file name
	Return	: None
*/
void Imitation::loadMapping(const string fileName) 
{	
	int i, num;
	string extRep, intRep;

	fstream fin;
	
	// read mapping (external/internal representation) from a file
	fin.open (fileName.c_str(), ios::in);
	fin >> num;
	for (i=0; i<num; ++i)
	{
		fin >> extRep;
		fin >> intRep;
		mMap.insert(make_pair(extRep, intRep));
	}
	fin.close();
}

/*
	Function: loadNumMapping()
	Desc.	: load the numeric representation of observed/internal model
	Para.	: fileName, file name
			  numMap, variable used to store the mapping
	Return	: None
*/
map<string, double> Imitation::loadNumMapping(const string fileName)
{
	int i, num;
	double numRep;
	string extRep;

	map<string, double> numMap;

	fstream fin;

	// read number representation of external state
	fin.open (fileName.c_str(), ios::in);
	fin >> num;
	for (i=0; i<num; ++i)
	{
		fin >> extRep;
		fin >> numRep;
		numMap.insert(make_pair(extRep, numRep));
	}
	fin.close();

	return numMap;
}

/*
	Function: loadNewDemos()
	Desc.	: load new demonstrations
	Para.	: fileName, file name where new demonstrations are stored
	Return	: None
*/
void Imitation::loadNewDemos(string fileName)
{
	string object, sep;
	fstream fin;

	int i, numOfDemo;
	
	// clear variable	
	newDemos.clear();
	newImits.clear();

	// open file
	fin.open(fileName.c_str(), ios::in);

	fin >> numOfDemo;
	for (i=0; i<numOfDemo; ++i)
	{
		// ignore *, demonstration separator
		fin >> sep;

		// load demonstration
		newDemos.push_back(readObservedModel(fin));
	}
	fin.close();
}

/*
	Function: generateSamples()
	Desc.	: Generate sample data from single-action demonstration, which is used to initialize the neural network
	Para.	: None
	Return	: None
*/
void Imitation::generateSamples()
{
	size_t i, j;
	State s1, s2;

	// prepare training data, using first demonstration
	samples.clear();
	for (i=0; i<newDemos.size(); ++i)
	{
		// set observed model
		setCurrentObservedModel(newDemos[i].objects, newDemos[i].states);
		for (j=0; j<newDemos[i].states.size(); ++j)
		{
			// for each state, make a observed/internal pair
			s1 = newDemos[i].states[j];
			s2 = mapping(s1,mMap);

			samples.push_back(convert(s1,s2));
		}
	}
}

/*
	Function: setCurrentObservedModel()
	Desc.	: set one demonstration as current observed model
	Para.	: observedObjects, all the object in current demonstration
			  observedStates, the sequence of states in current demonstration
	Return	: None
*/
void Imitation::setCurrentObservedModel(const vector<Object> &observedObjects, const vector<State> &observedStates)
{
	size_t i;

	currObservedObjects = observedObjects;
	currObservedStates = observedStates;

	// set internal objects
	intObjects.clear();
	for (i=0; i<observedObjects.size(); ++i)
		intObjects.push_back(mapping(observedObjects[i], mMap));
}

/*
	Function: setCurrentObservedModel()
	Desc.	: set one demonstration as current observed model
	Para.	: observedObjects, all the object in current demonstration
			  observedStates, the sequence of states in current demonstration
			  internalObjects, all the objects in the imitation environment
	Return	: None
*/
void Imitation::setCurrentObservedModel(const vector<Object> &observedObjects, const vector<State> &observedStates, const vector<Object> internalObjects)
{
	currObservedObjects = observedObjects;
	currObservedStates = observedStates;
	intObjects = internalObjects;
}

/*
	Function: testing()
	Desc	: Given a demonstration, see if the imitator can successully imitate
	Para	: fileName, the name of the file where the demonstration is stored.
	Return	: None
*/
void Imitation::testing(int modelState)
{
	string outputFile;
	fstream fin, fout;
	fin.open(NNFILE.c_str());

	// check whether the network is created or not
	if (fin.is_open())
		nn.create(NNFILE);
	else
	{
		cout << "The network doesn't exist!" << endl;
		fin.close();
		return;
	}
	fin.close();

	switch (modelState)
	{
		case EXPLOITATION:
			outputFile = "test_0.txt";
			break;
		case HANDCODE_EW:
			outputFile = "test_1.txt";
			break;
		case HANDCODE:
			outputFile = "test_2.txt";
	}
	fout.open(outputFile.c_str(), ios::out);
	
	test_1(modelState, fout);
	test_N(modelState, fout);

	// load observed model for testing
	loadNewDemos("TESTING.txt");
	loadImitObjects("TESTING_IMIT.txt");
	test_new(modelState, fout);

	fout.close();
}
void Imitation::test_new(int modelState, fstream &fout)
{
	double reward;
	psType policySibling;
	
	//cout << "3rd: ";
	fout << "3rd new testing result: " << endl;
	for (int i=0; i<newDemos.size(); ++i)
	{
		// set current observed model
		setCurrentObservedModel(newDemos[i].objects, newDemos[i].states, imitObjects[i]);

		// the demo is same as training, but the objects are different
		//switch (newDemos[i].num)
		//{
		//	case 0:		// trash cleaning
		//		currObservedObjects[0].color = "GREEN";
		//		currObservedObjects[0].texture = "PAPER";
		//		break;
		//	case 1:		// toy collection
		//		currObservedObjects[0].color = "BLUE";
		//		currObservedObjects[0].texture = "PLASTIC";
		//		break;
		//	case 301:
		//		intObjects[0].texture = "STONE";
		//		intObjects[1].texture = "METAL";
		//}

		// run the A* algorithm
		policySibling = AStarSearch(modelState, currAStarTree);
		
		// calculate reward
		reward = calcReward(policySibling.first, newDemos[i].num);
		//cout << "task " << i << " reward: " << reward << endl;		
		cout << reward << " ";
		fout << "task " << i << " reward: " << reward << endl;
		
		// output policy
		printNodes(fout, policySibling.first);
		fout << "sibling nodes: " << endl;
		printNodes(fout, policySibling.second);
	}
	cout << endl;
}
void Imitation::test_1(int modelState, fstream &fout)
{
	int i;
	double reward, totReward;

	psType policySibling;

	// testing single action's tasks
	loadNewDemos("1-ActionObservedModel.txt");
	//cout << "1st: ";
	fout << "1st-step training result: " << endl;

	totReward = 0;
	for (i=0; i<newDemos.size(); ++i)
	{
		// load observed model
		setCurrentObservedModel(newDemos[i].objects, newDemos[i].states);
				
		// run the A* algorithm
		policySibling = AStarSearch(modelState, currAStarTree);

		// calculate reward
		reward = calcReward(policySibling.first, newDemos[i].num);
		totReward += reward;

		// output policy
		cout << reward << " ";
		fout << "task: " << i << " reward: " << reward << endl;
		printNodes(fout, policySibling.first);
	}

	cout << totReward/newDemos.size() << " ";
	fout << "1st-step training - total reward: " << totReward/newDemos.size() << endl;
}
void Imitation::test_N(int modelState, fstream &fout)
{
	int i, j;
	double reward;

	psType policySibling;

	loadNewDemos("observedModel.txt");

	//cout << "2nd: ";
	fout << "2nd-step training result: " << endl;
	double totReward = 0;
	int nCount = 0;
	for (i=0; i<newDemos.size(); ++i)
	{
		nCount += imitationEnv[newDemos[i].num].size();
		for (j=0; j<imitationEnv[newDemos[i].num].size(); j++)
		{
			// load observed model
			setCurrentObservedModel(newDemos[i].objects, newDemos[i].states);

			changeImitationEnvironment(newDemos[i].num, j);
				
			// run the A* algorithm
			policySibling = AStarSearch(modelState, currAStarTree);

			// calculate reward
			reward = calcReward(policySibling.first, newDemos[i].num);
			totReward += reward;		

			// output policy
			cout << reward << " ";
			fout << "task " << i << " reward: " << reward << endl;
			printNodes(fout, policySibling.first);
			fout << "sibling nodes: " << endl;
			printNodes(fout, policySibling.second);
		}
	}
	cout << totReward/nCount << " ";
	fout << "2st-step training - total reward: " << totReward/nCount << endl;
}
void Imitation::generatePSFromNN()
{
	size_t i, j;
	double reward;
	psType policySibling;

	fstream fin;
	fin.open(NNFILE.c_str());

	// check whether the network is created or not
	if (fin.is_open())
		nn.create(NNFILE);
	else
	{
		cout << "The network doesn't exist!" << endl;
		fin.close();
		return;
	}
	fin.close();

	// testing single action's tasks
	loadNewDemos("1-ActionObservedModel.txt");

	vector<InternalModel> imits;
	InternalModel im;
	
	for (i=0; i<newDemos.size(); ++i)
	{
		imits.clear();
		for (j=0; j<imitationEnv[newDemos[i].num].size(); j++)
		{
			// load observed model
			setCurrentObservedModel(newDemos[i].objects, newDemos[i].states);

			changeImitationEnvironment(newDemos[i].num, j);
					
			// run the A* algorithm
			policySibling = AStarSearch(EXPLOITATION, currAStarTree);
			im.policy = policySibling.first;
			reward = calcReward(policySibling.first, newDemos[i].num);
			//im.policy[0].f =99;
		
			imits.push_back(InternalModel(intObjects, policySibling.first, policySibling.second, reward));
		}
		newImits.push_back(imits);
	}
	save();
}
/*
	Function: learning()
	Desc	: Given a set of demonstration, the imitator try to imitate
	Para	: None
	Return	: None
*/
void Imitation::learning(string fileName)
{
	fstream fin;
	
	fin.open(NNFILE.c_str());
	// check whether the network is created or not
	if (fin.is_open())
		//load parameters from file
		nn.create(NNFILE);
	else
	{
		// load single-action demos
		loadNewDemos("1-ActionObservedModel.txt");

		// generate samples
		generateSamples();

		// initialize neural network
		nn.create(NUM_OF_INPUT, numOfHiddenUnits, samples);		// 10
		nn.save("nn_0.txt");
		// parameters for single-action demos
		stdDeviation = INIT_STD_DEV;

		// train with single-action demos
		training();

		// save network
		nn.save("nn_1.txt");

		// save learned demos
		save();

		// store learned task
		learnedDemos.insert(learnedDemos.begin(), newDemos.begin(), newDemos.end());
		learnedImits.insert(learnedImits.begin(), newImits.begin(), newImits.end());

		/*fin.close();
		return;*/
	}
	fin.close();

	// load multi-action demos
	loadNewDemos(fileName);

	// parameters for multi-action demos
	stdDeviation = INIT_STD_DEV;

	// train with multi-action demos
	training();

	// save network
	nn.save("nn_2.txt");
	
	// save learned demos
	save();
}

void Imitation::testAction(State& s, int iAction, string p1, string p2)
{
	Action a = actions[iAction];
	a.parameterize(p1, p2);

	s = a.Execute(s);
	cout << s.toString();
	cout << s.nextToObjects() << endl;
}

/*
	Function: getPolicy()
	Desc.	: Generate policy starting from goal state
	Para.	: lBackpropagate, whether the cost is backpropagated up or not
	Return	: a sequnce of states generated by A* algorithm
*/
psType Imitation::getPolicy(const tree<Node>& aStarTree, bool lBackpropagate)
{
	list<Node> policy, siblings;

	tree<Node>::sibling_iterator sIter;

	// get the first node on the open list, which is the goal state
	treeNode *pre = openList.front();
	double childCost = pre->data.f;
	while(pre != 0)
	{
		// except the root node, each policy node's f is updated by max(itself, min(itchildren))
		if (lBackpropagate && pre->parent!=0 && childCost > pre->data.f)
			policy.push_front(Node(pre->data.state, pre->data.level, pre->data.g, childCost - pre->data.g));
		else
		{
			childCost = pre->data.f;
			policy.push_front(Node(pre->data.state, pre->data.level, pre->data.g, pre->data.h));
		}
		
		//find its sibling
		if (pre->parent != 0)
		{
			for (sIter = aStarTree.begin(pre->parent); sIter != aStarTree.end(pre->parent); ++sIter)
				if (*sIter != pre->data)
					siblings.push_front(*sIter);
		}
		pre= pre->parent;
	}

	return make_pair(vector<Node>(policy.begin(), policy.end()), vector<Node>(siblings.begin(), siblings.end()));
}

/*
	Function: addToList()
	Desc.	: Add a node into a list, its position is based on state's cost
	Para.	: tree, the state list which is sorted in ascent order on cost
			  node, an internal state that is going to be add to the tree
	Return	: None
*/
void Imitation::addToList(list<treeNode *> &target, treeNode *node)
{
	list<treeNode *>::iterator p;

	for (p=target.begin(); p != target.end(); ++p) 
		// compare the cost
		if ((*p)->data.f > node->data.f)
			break;

	// insert new state
	target.insert(p, node);
}

/*
	Function: load()
	Desc	: read information regarding to the demonstrations that have been learned and load them
	Para	: None
	Return	: None
*/
void Imitation::load()
{
	int i, num;
	string fileName;

	fstream fin;
	fin.open("imitation.txt", ios::in);

	// num of demonstrations that have been learned
	fin >> num;
	for (i=0; i<num; ++i)
	{
		// read the file's name
		fin >> fileName;

		// load each demonstration and its imitation
		loadLearnedDemo(fileName);
	}
	fin.close();
}

/*
	Function: save()
	Desc	: save all the demonstrations that have been learned
	Para	: string, file name
	Return	: None
*/
void Imitation::save()
{
	int i, numOfDemo;
	fstream fout;

	fout.open("imitation.txt", ios::out);

	// total number of demonstration that have been learned
	numOfDemo = learnedDemos.size() + newDemos.size();

	fout << numOfDemo << endl;
	// save new demonstrations
	for (i=0; i<numOfDemo; ++i)
		fout << "demo" << i << ".txt" << endl;

	fout.close();
	
	// save each demonstration separately
	saveLearnedDemos();
}

/*
	Function: saveLearnedDemos()
	Desc	: save all demonstrations learned
	Para	: None
	Return	: None
*/
void Imitation::saveLearnedDemos()
{
	size_t i, j;
	int num;
	string fileName;
	psType policySibling;

	fstream fout;

	num = learnedDemos.size();
	// the previous learned task won't change, but how about the total number of the demonstration
	for (i=0; i<newDemos.size(); ++i)
	{
		fileName = "demo" + convertToString(num) + ".txt";

		// open the output file
		fout.open(fileName.c_str(), ios::out);

		// save demonstration
		fout << newDemos[i].toString();

		// num of imitation
		fout << newImits[i].size() << endl;
		// save imitation
		for (j=0; j<newImits[i].size(); ++j)
		{
			// load observed model
			setCurrentObservedModel(newDemos[i].objects, newDemos[i].states, newImits[i][j].objects);

			// generate the current policy and calculate the reward
			policySibling = AStarSearch(EXPLOITATION, currAStarTree);
			newImits[i][j] = InternalModel(intObjects, policySibling.first, policySibling.second);
			/*newImits[i][j] = AStarSearch(EXPLOITATION);*/
			
			// recalculate reward
			newImits[i][j].reward = calcReward(newImits[i][j].policy, newDemos[i].num);
		
			// imitation separator
			fout << "#" << endl;
			
			// imitation detail
			fout << newImits[i][j].toString();
		}

		fout.close();
		num++;
	}
}

/*
	Function: calcReward()
	Desc	: Calculate reward for a given policy
	Para	: policy generated by A* algorithm
			  numOfDemo, demonstration's number. default is -1 for single-action training
	Return	: double, reward calculated
*/
double Imitation::calcReward(vector<Node> &policy, int numOfDemo)
{
	int i, iAction;
	double reward;

	vector<Node>::iterator iter, goalIter;

	State goalState, state;

	reward =0;
	goalIter = policy.end();

	// goal state in current observed states
	goalState = mapping(currObservedStates[currObservedStates.size()-1], mMap);

	// since the first state is the start state, no action will be taken, this state should be excluded from reward calculation
	for (iter = policy.begin()+1; iter!=policy.end(); ++iter)
	{
		iAction = iter->state.action;
		// penalty
		if (iAction != -1)
			reward -=actions[iAction].cost;

		// check if the current state is same as goal state in the demonstration, don't give any reward anymore
		if (iter->state.state == goalState)
			goalIter = iter;
	}

	if (currObservedStates.size() == 2)					// single action task
	{
		if (goalIter != policy.end())
			reward += 20;
	}
	else												// multi action task
	{
		if (goalIter != policy.end())
			state = goalIter->state.state;				// goal state
		else
			state = (--goalIter)->state.state;			// final state

			// check current action's number
		switch (numOfDemo) 
		{
			case TRASH_CLEANING:
				reward += calcRewardForCleaning(state);
				break;
			case TOY_COLLECTION:
				reward += calcRewardForCollection(state);
				break;
			case 301:
				reward += calcRewardForDoubleDrop(state);
				break;
			case FUTON_MATCH_1:
			case FUTON_MATCH_2:
			default:
				reward += calcReward(state);
				break;
		}
	}

	return reward;
}
/*
	Function: calcRewardForCleaning()
	Desc.	: calculate the reward for cleaning task
	Para.	: state, the state that either match the final observed state or the final internal state
	Retur	: reward for this task
*/
double Imitation::calcRewardForCleaning(State state)
{
	double reward = 0;
	string s = state.toString();
	
	if (s.find("NEXT Imitator Trashcan") != -1 && s.find("ON Gripper NULL") != -1 && s.find("ObjA") == -1)
		reward = 80;
	else
		if (s.find("NEXT Imitator Trashcan") != -1 && s.find("ON Gripper ObjA") != -1)
			reward = 60;
		else
			if (s.find("ON Gripper ObjA") != -1 ||
				(s.find("NEXT Imitator Trashcan") != -1 && s.find("NEXT Imitator ObjA") != -1))
				reward = 40;
			else
				if (s.find("NEXT Imitator ObjA") != -1)
					reward = 20;

	return reward;
}

double Imitation::calcRewardForDoubleDrop(State state)
{
	double reward = 0;
	string s = state.toString();
	
	if (s.find("ObjA") == -1 && s.find("ObjB") == -1)
		reward = 160;
	else
	{
		if (s.find("ObjA") == -1 || s.find("ObjB") == -1)
			reward += 80;

		if (s.find("NEXT Imitator Trashcan") != -1 && 
			(s.find("ON Gripper ObjA")!=-1 || s.find("ON Gripper ObjB")!=-1))
			reward += 60;
		else
			if (s.find("ON Gripper ObjA") != -1 || s.find("ON Gripper ObjB")!=-1)
				reward += 40;
			else
				if (s.find("NEXT Imitator ObjA") != -1 || s.find("NEXT Imitator ObjB") != -1)
					reward += 20;
	}
	return reward;
}
double Imitation::calcReward(State state)
{
	size_t i;

	double reward = 0;
	for (i=0; i<currObservedStates.size(); ++i)
		if (state == mapping(currObservedStates[i], mMap))
		{
			reward = i*20;
			break;
		}

	return reward;
}

double Imitation::calcRewardForCollection(State state)
{
	double reward = 0;
	string s = state.toString();

	if (state == mapping(currObservedStates[currObservedStates.size()-1], mMap))
		reward = 80;
	else if (s.find("NEXT Imitator ToyCorner") != -1 && s.find("ON Gripper Toy") != -1)
		reward = 60;
	else if (s.find("ON Gripper Toy") != -1)
		reward = 40;
	else if (state.findPattern(Relation("NEXT", "Imitator", "Toy")) != state.end())
		reward = 20;

	return reward;
}

/*
	Function: saveData()
	Desc.	: save input/expected output into a file
	Para.	: fileName, file Name
	Return	: None
*/
void Imitation::saveData(string fileName)
{
	size_t i, j;
	fstream fout;

	fout.open(fileName.c_str(), ios::out);

	// total number of training data
	fout << inputs.size() << " " << inputs[0].size() << endl;
	for (i=0; i<inputs.size(); ++i)
	{
		// inputs
		for (j=0; j<inputs[i].size(); ++j)
			fout << inputs[i][j] << " ";

		// expected output
		fout << expectedOutputs[i] << endl;
	}
	fout.close();
}

/*
	Function: loadData()
	Desc.	: load input/expected output from a file
	Para.	: fileName, file Name
	Return	: None
*/
void Imitation::loadData(string fileName)
{	
	int i, j, numOfSample, numOfInput;	
	double val;

	fstream fin;

	vector<double> input;

	fin.open(fileName.c_str(), ios::in);
	// if the file does not exist, exit
	if (!fin.is_open())
	{
		fin.close();
		return;
	}

	inputs.clear();
	expectedOutputs.clear();

	// read total number of training data
	fin >> numOfSample;
	// # of input in each training data
	fin >> numOfInput;
	for (i=0; i<numOfSample; ++i)
	{
		input.clear();
		for (j=0; j<numOfInput; ++j)
		{
			// input
			fin >> val;
			input.push_back(val);
		}
		inputs.push_back(input);

		// expected output
		fin >> val;
		expectedOutputs.push_back(val);
	}
	fin.close();
}

/*
	Function: AStarSearch
	Desc.	: Using A* algorithm to find a policy
	Para.	: modelState, EXPLORATION (policy exploration) or EXPLOITATION
	Return	: a pair, which including policy and sibling generated by A* algorithm
*/
psType Imitation::AStarSearch(int modelState, tree<Node>& aStarTree)
{
	double newCost;
	bool findSuccessor;

	tree<Node>::pre_order_iterator treeIter, parentIter;

	// sequence of states selected by the internal model
	list<InternalState> successors;
	list<InternalState>::iterator successorIter;

	treeNode *currState;
	list<treeNode *>::iterator p;

	psType policySiblings;

	// clear the open/closed list
	openList.clear();
	closedList.clear();
	aStarTree.clear();

	// start state is the first state in the current observed model
	InternalState startState = InternalState(-1,mapping(currObservedStates[0], mMap),0);

	// initialize the nextTo property
	startState.state.updateNextTo();

	// calculate distance for start state
	switch (modelState)
	{
		case HANDCODE:
			startState.distance = handCode(currObservedStates[startState.extStateNum], startState.state);
			break;
		case HANDCODE_EW:
			startState.distance = simpleDistance(currObservedStates[startState.extStateNum], startState.state);
			break;
		default:
			startState.distance = calcDistance(currObservedStates[startState.extStateNum], startState.state, EXPLOITATION);
	}
		
	// create a node for start state
	Node newNode = Node(startState, 0, startState.distance, calcHeuristicCost(startState));
	treeIter = aStarTree.set_head (newNode);
	
	// put start node into open list
	openList.push_back(treeIter.node);
	
	currState = openList.front();
	// loop until the first state in open list correspond to the last observed state
	while (!isGoalState(currState->data.state)) 
	{
		// find its location in the A* tree
		parentIter = find(aStarTree.begin(), aStarTree.end(), currState->data);

		findSuccessor = false;

		// remove current state from open list
		openList.erase(openList.begin());

		// generate sucessors of current state
		successors = currState->data.state.genSuccessors(actions, intObjects);

		// choose a successor
		if (modelState == EXPLORATION)
			chooseASuccessor(successors);

		// handle each successor
		for (successorIter=successors.begin(); successorIter != successors.end(); ++successorIter) 
		{
			switch (modelState)
			{
				case HANDCODE:
					successorIter->distance = handCode(currObservedStates[successorIter->extStateNum], successorIter->state);
					break;
				case HANDCODE_EW:
					successorIter->distance = simpleDistance(currObservedStates[successorIter->extStateNum], successorIter->state);
					break;
				default:
					// calculate difference between observed state and internal state, update gVal
					successorIter->distance = calcDistance(currObservedStates[successorIter->extStateNum], successorIter->state, EXPLOITATION);
			}

			// calculate new g
			newCost = currState->data.g + calcActionCost(successorIter->action) + successorIter->distance;
			newNode = Node((*successorIter), currState->data.level + 1, newCost, calcHeuristicCost((*successorIter)));
			
			//remove from open list that has higher cost 
			treeIter = find(aStarTree.begin(), aStarTree.end(), newNode);
			if (treeIter != aStarTree.end())
			{
				// skip when exists on the open list which has less cost
				if (newNode.g >= treeIter->g)
					continue;

				// remove the node and its children from open/closed list
				removeFromList(aStarTree, treeIter.node);

				// remove node from A* tree
				aStarTree.erase(treeIter);
			}
			
			treeIter = aStarTree.append_child(parentIter, newNode);
			if (modelState != EXPLORATION)
				// add successor to the open list, the position is determined by the cost
				addToList(openList, treeIter.node);
			else
				if (findSuccessor)
					// its siblings are appended at the back of the open list
					openList.push_back(treeIter.node);
				else
				{
					findSuccessor = true;

					// the first successor will be inserted in the front of the open list
					openList.insert(openList.begin(), treeIter.node);
				}
		}

		// Add current node to closed list, just add the back
		closedList.push_back(currState);

		// get next node on the open list
		currState = openList.front();
	}

	// get the policy and siblings
	policySiblings = getPolicy(aStarTree);

	backpropagateHeuristicCost(aStarTree);

	// cleanup open/closed list
	openList.clear();
	closedList.clear();

	return policySiblings;
}

void Imitation::removeFromList(const tree<Node>& aStarTree, treeNode *node)
{
	// remove itself from open/closed list
	openList.erase(remove(openList.begin(), openList.end(), node), openList.end());
	closedList.erase(remove(closedList.begin(), closedList.end(), node), closedList.end());
	
	if (node->first_child !=0)
		// remove its children from open/closed list
		for (tree<Node>::sibling_iterator siblingIter = aStarTree.begin(node); siblingIter != aStarTree.end(node); ++siblingIter)
			removeFromList(aStarTree, siblingIter.node);
}
/*
	Function: chooseASuccessor()
	Desc	: Reorder the successors based on a distance exploration
	Para	: All the successor of current node in the A* tree
	Return	: None
*/
void Imitation::chooseASuccessor(list<InternalState> &successors)
{
	list<InternalState>::iterator iter;

	for (iter = successors.begin(); iter!=successors.end(); ++iter)
		// random generate a distance for each successor based on its mean and variance
		iter->distance = calcDistance(currObservedStates[iter->extStateNum], iter->state, EXPLORATION);

	// reorder the successors based on its distance
	successors.sort();
}

/*
	Function: batchUpdate
	Desc	: Execute a batch update
	Para	: None
	Return	: None 
	Note	: the distance of each state-pair is stored in exploredOutput, which including the number and the sum of distance
*/
void Imitation::batchUpdate(double totRewardDiff)
{
	int i, j, iCount;
	double currOutput, expectedOutput, maxRewardDiff, err;

	bool samePolicy;

	vector<double> input, currOutputs;

	for (iCount=0; iCount<LEARNING_ROUND; ++iCount)
	{
		// update network when it does not produce same policy
		if (DEBUG_MODE)
		{
			fout_update << "round: " << iCount << endl;
			fout_policy << "after round: " << iCount << endl;
			fout_update << "expected output" << setw(15) << "#" << endl;
		}
	
		maxRewardDiff = *max_element(rewardDiffs.begin(), rewardDiffs.end());

		cout << "total reward difference: " << maxRewardDiff << endl;
		
		vector<vector<double> >::iterator inputIter = inputs.begin();
		vector<vector<double> >::iterator outputIter = exploredOutputs.begin();
		vector<double>::iterator rewardDiffIter = rewardDiffs.begin();

		while (inputIter != inputs.end())
		{
			/*if ((*outputIter)[1] == 0)
			{
				inputs.erase(inputIter);
				exploredOutputs.erase(outputIter);
				rewadDiff.erase(rewardDiffIter);

				continue;
			}*/

			if (DEBUG_MODE)
				fout_update << (*outputIter)[0] << setw(15) << (*outputIter)[1] << setw(15) << (*rewardDiffIter) << endl;
			
			expectedOutput = (*outputIter)[0] + (*outputIter)[1]/maxRewardDiff;

			if (expectedOutput<0)
				expectedOutput = 0;

			expectedOutputs.push_back(expectedOutput);

			currOutput = calcDistance((*inputIter));
			currOutputs.push_back(currOutput);
			
			++inputIter;
			++outputIter;
			++rewardDiffIter;
		}

		// using scaled conjugate gradient algorithm to update weights and bias
		cout << "round: " << iCount << endl;
		err = nn.scaledConjugateGradient(inputs, expectedOutputs, 1e-005, BATCH_UPDATE_ITERATION);

		if (DEBUG_MODE)
		{
			fout_update << "Err: " << err << endl;
			fout_update << "old" << setw(15) << "expected" << setw(15) << "updated" << setw(15) << "diff" << endl;
			fout_err << "round: " << iCount << " Err: " << err << endl;
		
			for (i=0; i<inputs.size(); ++i)
			{
				currOutput = calcDistance(inputs[i]);
				fout_update << currOutputs[i] << setw(15) << expectedOutputs[i] << setw(15) << currOutput << setw(15) << expectedOutputs[i] - currOutput << endl;
			}
		}

		// clear the training set
		currOutputs.clear();
		clearTrainingSet();

		samePolicy = true;
		totRewardDiff = 0;
		// update distance
		for (i=0; i<newDemos.size(); ++i)
		{
			j = idxOfCurrImits[i];
			if (!policyIsSatisfied(newDemos[i], newImits[i][j]))
			{
				samePolicy = false;
				totRewardDiff += newImits[i][j].reward-currReward;
			}
		}

		// consider previously learned demo
		for (i=0; i<learnedDemos.size(); ++i)
			for (j=0; j<learnedImits[i].size(); ++j)
				if (!policyIsSatisfied(learnedDemos[i], learnedImits[i][j]))
				{
					samePolicy = false;
					totRewardDiff += learnedImits[i][j].reward-currReward;
				}

		// return if the current network produces the same policy
		if (samePolicy)
			break;		
	}
	fout_err << "round: " << iCount << " Err: " << err << endl;
}

/*
	Function: policyIsSatisfied()
	Desc	: Given a demo, check whether its imitation is satisfied or not
	Para.	: demo, a demonstration
			  imit, the imitation strategy for the demo
	Return	: true when the imitation's policy is satisfied; otherwise return false
*/
bool Imitation::policyIsSatisfied(ObservedModel &demo, InternalModel &imit)
{
	// set current observed model
	setCurrentObservedModel(demo.objects, demo.states, imit.objects);

	// recalculate distance
	recalcDistance(imit);
	
	// call A* search algorithm, each A* tree for each task
	currPolicySibling = AStarSearch(EXPLOITATION, currAStarTree);
	
	// calculate reward for current policy
	currReward = calcReward(currPolicySibling.first, demo.num);	
	if (imit.reward > currReward)
	{
		if (DEBUG_MODE)
		{
			fout_policy << "current policy after update: " << endl;
			printNodes(fout_policy, currPolicySibling.first);

			fout_policy << "new policy after update: " << endl;
			printNodes(fout_policy, imit.policy);
		}
		
		generateDistance(imit, imit.reward - currReward);
	}
	else
	{
		// update with current model which has higher reward
		imit.reward = currReward;
		imit.policy = currPolicySibling.first;
		imit.siblings = currPolicySibling.second;
		
		generateDistance(imit);
	}

	return (currReward >= imit.reward);
}
/*
	Function: addToTrainingSet
	Desc	: Add input/expected output into training set
	Para	: state, observed/internal state pair
			  output, current output
			  rewardDiff, reward difference for the task which this state pair appears
			  delta=0, the expected delta * reward difference
	Return	: None
*/
void Imitation::addToTrainingSet (const InternalState& state, double output, double rewardDiff, double delta)
{
	int iPos;

	vector<double> input, expectedOutput;

	// convert state pair into numeric representation
	input = convert(currObservedStates[state.extStateNum], state.state);

	// check whether the same input exists already
	iPos = search(inputs, input);
	if (iPos != -1)
	{
		exploredOutputs[iPos][1] += delta;
		rewardDiffs[iPos] += rewardDiff;
	}
	else
	{
		inputs.push_back(input);

		// current output and its delta (it expected delta * rewardDiff)
		expectedOutput.push_back(output);
		expectedOutput.push_back(delta);
		rewardDiffs.push_back(rewardDiff);
		
		exploredOutputs.push_back(expectedOutput);
	}
}

/*
	Function: training()
	Desc.	: Take a set of demonstrations and train the network to produce optimal policy
	Para.	: None
	Return	: None
*/
void Imitation::training()
{
	bool unChanged;
	int i, j, iCount, iCountUnchanged;
	double newReward, totRewardDiff;

	// stream interface
	fstream fout_AStar, fout_solution;
	
	// new internal model
	psType newPolicySibling;
	InternalModel newIntModel;

	// imitation
	vector<InternalModel> imits;
	vector<InternalModel>::iterator p;
	
	// output stream for policy
	fout_solution.open("o_solution.txt", ios::out);

	// output stream for A* star tree
	fout_AStar.open("o_astar.txt", ios::out);

	// after each run, reward/penalty will be given, this is provided by the user or calculation
	iCount = 0;
	iCountUnchanged = 0;
	while (true)
	{
		// check if key 'x' is press or no more change happend during last 6000 iterations, if yes, exit loop
		//if ((kbhit() && (char)getch() == 'x') || iCountUnchanged>=6000)
		//	break;
		// exit when no more change happend during last 6000 iterations, if yes
		if (iCountUnchanged>=6000)
			break;

		cout << "round: " << iCount << " " << iCountUnchanged << " std. Dev: " << stdDeviation << endl;
		fout_rew << setw(10) << iCount << setw(10) << iCountUnchanged << endl;
		fout_oldRew << setw(10) << iCount << setw(10) << iCountUnchanged << endl;
		if (DEBUG_MODE)
		{
			fout_solution << endl << "round: " << iCount << " " << iCountUnchanged << " std. Dev: " << stdDeviation << endl;
			fout_AStar << endl << "round: " << iCount << " " << iCountUnchanged << " std. Dev: " << stdDeviation << endl;
		}
		unChanged = true;
		totRewardDiff = 0;
		clearTrainingSet();
		idxOfCurrImits.clear();
		for (i=0; i<newDemos.size(); ++i)
		{
			// load observed model
			setCurrentObservedModel(newDemos[i].objects, newDemos[i].states);

			// change imitation environment for multi-action task
			if (currObservedStates.size() > 2)
			//if (newDemos[i].num != -1)
				changeImitationEnvironment(newDemos[i].num);

			// generate the current policy and calculate the reward
			currPolicySibling = AStarSearch(EXPLOITATION, currAStarTree);
			currReward = calcReward(currPolicySibling.first, newDemos[i].num);

			// output reward
			fout_oldRew << "task: " << i << setw(4) << currObservedObjects[0].color.substr(0,3) << setw(4) << currObservedObjects[0].texture.substr(0,3) << 
				setw(4) << intObjects[0].color.substr(0,3) << setw(4) << intObjects[0].texture.substr(0,3) << setw(4) << currReward << endl;
			
			// output current policy
			if (DEBUG_MODE)
			{
				fout_AStar << "task: " << i << " A* tree: "<< endl;
				printTree(fout_AStar, currAStarTree);

				fout_solution << "task: " << i << " old Reward: " << currReward << endl;
				printNodes(fout_solution, currPolicySibling.first);
			}

			// generate a new policy and calculate reward
			newPolicySibling = AStarSearch(EXPLORATION, newAStarTree);
			newReward = calcReward(newPolicySibling.first, newDemos[i].num);
			
			// output reward distribution
			fout_rew << "task: " << i << setw(4) << currObservedObjects[0].color.substr(0,3) << setw(4) << currObservedObjects[0].texture.substr(0,3) << 
				setw(4) << intObjects[0].color.substr(0,3) << setw(4) << intObjects[0].texture.substr(0,3) << setw(4) << newReward << endl;

			if (DEBUG_MODE)
			{
				fout_solution << "task: " << i << " new Reward: " << newReward << endl;
				printNodes(fout_solution, newPolicySibling.first, true);
			}
			cout << "task: " << i << " old reward: " << currReward << " new reward: " << newReward << endl << endl;

			// if the current policy is as good as the new one, go for next exploration
			if (newReward > currReward)
			{	
				unChanged = false;
				totRewardDiff += newReward - currReward;
				newIntModel = InternalModel(intObjects, newPolicySibling.first, newPolicySibling.second, newReward);
				generateDistance(newIntModel, newReward - currReward);
			}
			else
			{
				newIntModel = InternalModel(intObjects, currPolicySibling.first, currPolicySibling.second, currReward);
				generateDistance(newIntModel);
			}
			
			// save imitation environment for each demonstration (Don't save policy, it may change)
			if (iCount == 0 && iCountUnchanged == 0)
			{
				imits.clear();
			
				imits.push_back(newIntModel);
				newImits.push_back(imits);
				// the first one
				idxOfCurrImits.push_back(0);
			}
			else
			{
				// check whether the same imitation case appeared alreay
				for (j=0; j<newImits[i].size(); ++j)
					if (equal(intObjects.begin(), intObjects.end(), newImits[i][j].objects.begin()))
						break;

				// if not, insert new entries
				if (j != newImits[i].size())
					newImits[i][j] = newIntModel;
				else
					newImits[i].push_back(newIntModel);
				idxOfCurrImits.push_back(j);
			}
		}

		if (unChanged)
		{
			iCountUnchanged++;
			continue;
		}
		iCountUnchanged = 0;

		for (i=0; i<learnedDemos.size(); ++i)
			for (j=0; j<learnedImits[i].size(); ++j)
				policyIsSatisfied(learnedDemos[i], learnedImits[i][j]);

		// calculate distance based on the new policy and train the network
		batchUpdate(totRewardDiff);

		// update the standard deviation
		if (iCount%100 == 0)
			stdDeviation *= DISCOUNT_FACTOR;
	
		// save neural network configuration, debug purpose
		nn.save("nn_tmp.txt");

		iCount++;
	}
	
	fout_solution.close();
	fout_AStar.close();
}

void Imitation::printNodes(fstream &fout, const vector<Node>& nodes,  bool standardOutput)
{
	int i;
	vector<vector<string> > intSState;

	fout << "size: " << nodes.size() << endl;
	for (i=0; i<nodes.size(); ++i)
	{
		if (standardOutput)
			cout << nodes[i].toString();
		fout << nodes[i].state.extStateNum << " A: " << nodes[i].state.action << " D: " << nodes[i].state.distance;
		fout << " T: " << nodes[i].f << endl;
		
		//fout << nodes[i].toString();
		intSState = stateToString(nodes[i].state.state);
		for (int j=0; j<intSState.size(); ++j)
		{
			for (int k=0; k<intSState[j].size(); ++k)
				fout << intSState[j][k] << " ";
			fout << endl;
		}
	}
	fout << endl;
}

void Imitation::printTree (fstream &fout, const tree<Node>& aStarTree, bool standardOutput)
{
	
	tree<Node>::pre_order_iterator treeIter;

	if(!aStarTree.is_valid(aStarTree.begin())) 
		return;

	int rootdepth=aStarTree.depth(aStarTree.begin());

	fout << "size: " << aStarTree.size() << endl;
	for (treeIter = aStarTree.begin(); treeIter != aStarTree.end(); ++treeIter)
	{
		for(int i=0; i<aStarTree.depth(treeIter)-rootdepth; ++i) 
			fout << "--";

		if (standardOutput)
			cout << treeIter->toString();
		fout << "total cost: " << treeIter->f << endl;
		fout << treeIter->toString();
	}
	fout << endl;
}

/*
	Function: loadLearnedDemo()
	Desc	: load one demonstration and its imitation
	Para	: fileName, file name
	Return	: None
*/
void Imitation::loadLearnedDemo(const string fileName)
{
	int i, numOfImitation;
	string s;
	fstream fin;
	
	// imitations for one demonstration
	vector<InternalModel> intModels;
	
	// open file
	fin.open(fileName.c_str(), ios::in);

	// load demonstration
	learnedDemos.push_back(readObservedModel(fin));

	// load imitation
	fin >> numOfImitation;
	for (i=0; i<numOfImitation; ++i)
	{
		// ignore imitation separator #
		fin >> s;

		// one imitation is created and store temporarily
		intModels.push_back(readInternalModel(fin));
	}

	// all imitations are loaded and store in learedImits
	learnedImits.push_back(intModels);

	fin.close();
}

/*
	Function: recalcDistance()
	Desc	: recalculate the distance and other costs for nodes in the A* tree 
	Para	: intModel, internal model
	Return	: None
*/
void Imitation::recalcDistance(InternalModel &intModel)
{
	int i, levelOfParent;
	double gOfParent;
	
	// update distance for those nodes in the policy
	for (i=0; i<intModel.policy.size(); ++i)
	{
		if (i==0)
			gOfParent = 0;
		else
			gOfParent = intModel.policy[i-1].g;
		
		recalcDistance(intModel.policy[i], gOfParent);
	}
	
	// update distance for those nodes in the A* tree (sibling)
	for (i=0; i<intModel.siblings.size(); ++i)
	{
		// parent's level
		levelOfParent = intModel.siblings[i].level-1;

		recalcDistance(intModel.siblings[i], intModel.policy[levelOfParent].g);
	}

	// backpropagate the heuristic cost
	backpropagateHeuristicCost(intModel.policy);
}

/*
	Function: recalcDistance()
	Desc	: recalculate the distance and other cost for a single node
	Para	: node, the node that you want to recalculate its cost
			  gOfParent, the g of its parent
	Return	: None
*/
void Imitation::recalcDistance(Node &node, double gOfParent)
{
	double newDistance;

	// recalculate its distance
	newDistance = calcDistance(currObservedStates[node.state.extStateNum], node.state.state, EXPLOITATION);

	// update distance
	node.state.distance = newDistance;

	// recalculate g = gOfParent + action's cost and it new distance
	node.g = gOfParent + calcActionCost(node.state.action) + newDistance;

	// for the policy nodes, h may be changed, recalculate
	node.h = calcHeuristicCost(node.state);

	// update f
	node.f = node.g + node.h;
}

/*
	Function: backpropagateHeuristicCost()
	Desc.	: backpropagate heuristic cost from goal state to start state
	Para.	: policy
	Return	: None
	Note	: after backpropagate, the second node has the maximum cost among all the policy nodes.
*/
void Imitation::backpropagateHeuristicCost(vector<Node> &policy)
{
	int i;

	// except the root node, each policy node's f is updated by max(itself, min(itchildren))
	for (i=policy.size()-1; i>1; --i)
		// compare current node with its parent
		if (policy[i].f > policy[i-1].f)
		{
			// update parent's heuristic cost
			policy[i-1].h = policy[i].f - policy[i-1].g;
			policy[i-1].f = policy[i-1].g + policy[i-1].h;
		}
}

/*
	Function: generateDistance()
	Desc.	: check whether the policy in this internal model is satisfied
	Para	: intModel, an internal model
			  satisfied, whether the given policy is satisfied
				true, prepare training set with new policy nodes and their siblings
				false, calculate the distance to produce the given policy
			  rewardDiff, the difference between new reward and current reward, default = 0
	Return	: None
*/
void Imitation::generateDistance(InternalModel &intModel, double rewardDiff)
{
	int i, level;
	vector<Node> aStarTree;
	vector<Node>::iterator iter;

	if (rewardDiff == 0)
	{
		// policy nodes in the new A* tree
		//for (iter = intModel.policy.begin(); iter != intModel.policy.end(); ++iter)
		for (iter = intModel.policy.begin()+1; iter != intModel.policy.end(); ++iter)	// skip root node
			addToTrainingSet(iter->state, iter->state.distance);
		
		// siblings in the new A* tree
		for (iter = intModel.siblings.begin(); iter != intModel.siblings.end(); ++iter)
			addToTrainingSet(iter->state, iter->state.distance);

		return;
	}

	// find out the first unmatched nodes between the current and new policy
	typedef pair<vector<Node>::iterator, vector<Node>::iterator> misMatchType;
	misMatchType misMatch = mismatch(currPolicySibling.first.begin(), currPolicySibling.first.end(), intModel.policy.begin(), mem_fun_ref(&Node::operator ==));

	int matchLevel = misMatch.second->level-1;
	// the maximum cost in the current policy, exclude matched nodes
	double currPolicyCost = misMatch.first->f;
	
	// the maximum cost in the new policy, exclude matched nodes
	double newPolicyCost = misMatch.second->f;

	cout << "new policy cost: " << newPolicyCost << " curr. policy cost:" << currPolicyCost << endl;

	// for current A* tree, only consider those nodes that have the maximum cost along each branch
	tree<Node>::pre_order_iterator treeIter = find(currAStarTree.begin(), currAStarTree.end(), (*--misMatch.first));
	for (tree<Node>::sibling_iterator siblingIter=currAStarTree.begin(treeIter); siblingIter != currAStarTree.end(treeIter); ++siblingIter)
	{
		// skip new policy node
		if ((*siblingIter) == (*misMatch.second))
			continue;

		// maximum cost along that branch
		double maxCost = siblingIter->f;
		
		// stop when the iterator pointer to next sibling or the end of iterator
		tree<Node>::pre_order_iterator subTreeIter = siblingIter;
		while(true)
		{
			aStarTree.push_back(subTreeIter.node->data);

			// find minimum child
			tree<Node>::pre_order_iterator minIter= min_element(currAStarTree.begin(subTreeIter), currAStarTree.end(subTreeIter));
			
			// check whether the minimum child has same cost as current node
			if (minIter == currAStarTree.end(subTreeIter) || minIter->f != subTreeIter->f)
				break;	// exit when current node's cost is the maximum

			subTreeIter = minIter;
		}
	}

	int posCount = 0;
	int negCount = 0;

	vector<Node>::iterator p;
	
	// minSibling, minimum cost of sibling nodes in each level and its previous level, 
	// i.e. minSibling[3] is the minimum cost among the sibling nodes in level 3, 2 and 1
	// initialize with current policy cost
	vector<double> minSibling(intModel.policy.size(), currPolicyCost);

	// new silbings
	for (iter = intModel.siblings.begin(); iter != intModel.siblings.end(); ++iter)
	{
		// delete sibling from current A* tree
		p= find(aStarTree.begin(), aStarTree.end(), (*iter));
		if (p != aStarTree.end())
			aStarTree.erase(p);
		
		// sibling's level in the A* tree
		level = iter->level;

		// calculate minimum in each level and its following level
		if (iter->f < minSibling[level])
			minSibling[level] = iter->f;

		// count siblings whose cost is lower than same level policy node (each policy node's f = max(itself, min(it's children))
		if (iter->f <= intModel.policy[level].f)
			++posCount;
	}
	// propagate the minimum cost downward, miniSibling[i] is the minimum cost among level 1, 2, ..., i-1, i
	// first level is root, ignore
	for (i = 1; i < minSibling.size()-1; ++i)
		minSibling[i+1] = min(minSibling[i], minSibling[i+1]);

	// new policy
	for (i=0; i<intModel.policy.size(); ++i)
	{
		// delete new policy node from current A* tree
		p = find(aStarTree.begin(), aStarTree.end(), intModel.policy[i]);
		if (p!=aStarTree.end())
			aStarTree.erase(p);

		// count how many new policy nodes need to be decreased, not including matched nodes
		if (i > matchLevel && intModel.policy[i].f >= minSibling[i])
			++negCount;
	}

	// count how many nodes in the current A* tree need to be increased
	for (i=0; i<aStarTree.size(); ++i)
		if (aStarTree[i].f <= newPolicyCost)
			++posCount;

	double diff, delta;

	// compare each node in the policy with minimum cost in current/new A* tree
	for (i=1; i<intModel.policy.size(); ++i)	// skip root node
	//for (i=0; i<intModel.policy.size(); ++i)
	{
		diff = intModel.policy[i].f - minSibling[i];
		if (i <= matchLevel || diff < 0)
			// no change for matched policy nodes or those whose cost f < minCost
			delta = 0;
		else
		{
			// this node'f >= minCost
			delta = -(diff/negCount * (1+ MARGIN_PER) + MARGIN) * rewardDiff;
			
		}

		// add to the training set
		addToTrainingSet(intModel.policy[i].state, intModel.policy[i].state.distance, rewardDiff, delta);
	}

	// compare each sibling in the new A* tree with new policy cost
	for (i=0; i<intModel.siblings.size(); ++i)
	{
		level = intModel.siblings[i].level;
		
		// compare with same level policy node's f, which is max(itself, min(it's children)
		diff = intModel.policy[level].f - intModel.siblings[i].f;
		if (diff < 0)
			// this node's f > policy cost
			delta = 0;
		else
			// this node's f <= policy cost
			delta = (diff/posCount * (1 + MARGIN_PER) + MARGIN) * rewardDiff;

		// add to the training set
		addToTrainingSet(intModel.siblings[i].state, intModel.siblings[i].state.distance, rewardDiff, delta);
	}

	// compare each node in the current A* tree with new policy cost
	for (i=0; i<aStarTree.size(); ++i)
	{
		diff = newPolicyCost - aStarTree[i].f;
		if (diff < 0)
			// this node's f > policy cost
			delta = 0;
		else
			// this node's f <= policy cost
			delta = (diff/posCount * (1 + MARGIN_PER) + MARGIN) * rewardDiff;

		// add to the training set
		addToTrainingSet(aStarTree[i].state, aStarTree[i].state.distance, rewardDiff, delta);
	}
}

/*
	Function: calcActionCost
	Desc.	: calculate action's cost
	Para.	: action's num
	Return	: action's cost
*/
double Imitation::calcActionCost(int num)
{
	if (num == -1)
		return 0;
	else
		return actions[num].cost;
}

/*
	Function: backpropagateHeuristicCost()
	Desc.	: get the whole A* tree, including the states that have been visited and unvisited
	Para.	: lBackpropagate, whether the cost is backpropagated up or not
	Return	: all the nodes in the A* tree
*/
void Imitation::backpropagateHeuristicCost(tree<Node>& aStarTree)
{	
	tree<Node>::iterator_base iter;
	tree<Node>::sibling_iterator siblingIter;

	// get all the leaf node first
	iter = aStarTree.begin();
	for (siblingIter = aStarTree.begin(iter); siblingIter != aStarTree.end(iter); ++siblingIter)
		minMax(aStarTree, siblingIter);
}

double Imitation::minMax(tree<Node>& aStarTree, tree<Node>::iterator_base& iter)
{
	if (iter.node->first_child ==0)
		return iter->f;

	tree<Node>::sibling_iterator siblingIter;

	double minChild = -1;
	// find minimum cost among its children
	for (siblingIter = aStarTree.begin(iter); siblingIter != aStarTree.end(iter); ++siblingIter)
	{
		double childCost = minMax(aStarTree, siblingIter);
		
		if (minChild == -1 || childCost < minChild)
			minChild = childCost;
	}

	if (iter->f < minChild)
	{
		iter->h = minChild - iter->g;
		iter->f = iter->g + iter->h;
	}
	return iter->f;
}
/*
	Function: changeImitationEnvironment()
	Desc	:
	Para.	: numOfDemo, the demonstration number
			  idxOfAttr, the index of attribute, this parameter will be pickup randomly when not provided or invalid
	Return	: None
*/
void Imitation::changeImitationEnvironment(int numOfDemo, int idxOfAttr)
{
	string newAttr;

	// make imitation envrionment different from demonstration, random choose an attribute
	if (idxOfAttr == -1 || idxOfAttr >= imitationEnv[numOfDemo].size())
		idxOfAttr = rand() % imitationEnv[numOfDemo].size();

	newAttr = imitationEnv[numOfDemo][idxOfAttr];
	switch (numOfDemo)
	{
		case TRASH_CLEANING:
			// cleaning task, change objA's texture attribute
			intObjects[0].texture= newAttr;		// assume the object A is the first one on the object list
			break;
		case TOY_COLLECTION:
			// Toy collection, change Toy's color, exclusive green which indicate it is trash in the cleaning task
			intObjects[0].color = newAttr;
			break;
		case FUTON_MATCH_1:
			// Futon-Sofa match 1, change the texture of futon1 and sofa1
			intObjects[0].texture = newAttr;
			intObjects[2].texture = newAttr;
			break;
		case FUTON_MATCH_2:
			// Futon-Sofa match 2, change the texture of futon1
			intObjects[0].texture = newAttr;
			break;
	}
}
void Imitation::clearTrainingSet()
{	
	inputs.clear();
	expectedOutputs.clear();
	exploredOutputs.clear();
	rewardDiffs.clear();
}

/*
	Function: mapping()
	Desc.	: convert this instance to the internal representation 
	Para.	: mapping between observed model and internal model
	return	: Object, internal representation of this instance
*/
Object mapping(const Object& o, map<string,string>& m)
{
	return Object(m[o.name], m[o.color], m[o.texture]);
}

/*
	Function: mapping()
	Desc.	: convert this instance to the internal representation 
	Para.	: extState, the external representation of the relation
	return	: Relation, internal state representation of the relation
*/
Relation mapping(const Relation& r, map<string,string>& m)
{
	return Relation(m[r.relation], m[r.objA], m[r.objB]);
}

/* 
	Function: mapping()
	Desc.	: convert this instance into internal representation
	Para.	: extStateNum, the number of external state
	return	: internal state representation
*/
State mapping(State s, map<string,string>& m)
{	
	State intState;

	const size_t N = s.size();
	
	for (size_t i=0; i<N; ++i)
		//convert each relation in the observed state
		intState.add(mapping(s[i], m));

	return intState;
}

/********************************************* Nonmember Functions **************************************************/
/*
	Function: readObject()
	Desc.	: read object info through a stream interface and return an Object
	Para.	: fin, a stream interface
	Return	: an Object
*/
Object readObject(fstream &fin)
{
	string name, color, texture;
	
	fin >> name;
	fin >> color;
	fin >> texture;

	return Object(name, color, texture);
}

/*
	Function: readRelation()
	Desc.	: read relation info through a stream interface and return a Relation
	Para.	: fin, a stream interface
	Return	: a Relation
*/
Relation readRelation(fstream &fin)
{
	string relation, objA, objB;

	fin >> relation;
	fin >> objA;
	fin >> objB;

	return Relation(relation, objA, objB);
}

/*
	Function: readState()
	Desc.	: read state info through a stream interface and return a State
	Para.	: fin, a stream interface
	Return	: a State
*/
State readState(fstream &fin)
{
	int i, numOfRelation;

	State state;

	// num of relation in each state
	fin >> numOfRelation;
	for (i=0; i<numOfRelation; ++i)
		state.add(readRelation(fin));

	return state;
}

/*
	Function: readInternalState()
	Desc.	: read internal state info through a stream interface and return an InteralState
	Para.	: fin, a stream interface
	Return	: an InternalState
*/
InternalState readInternalState(fstream &fin)
{
	int action, extStateNum;
	double distance;
	State state;

	fin >> action;
	fin >> distance;
	fin >> extStateNum;

	state = readState(fin);

	return InternalState(action, state, extStateNum, distance);
}

/*
	Function: readObservedModel()
	Desc.	: read observed model info (demo) through a stream interface and return an ObservedModel
	Para.	: fin, a stream interface
	Return	: an ObservedModel
*/
ObservedModel readObservedModel(fstream &fin)
{
	int i, num, numOfObjects, numOfState;

	vector<State> states;
	vector<Object> objects;

	// demonstration's number
	fin >> num;

	// objects in the demonstration
	fin >> numOfObjects;

	// sequence of states in the demonstration
	fin >> numOfState;

	// objects' detail
	for (i=0; i<numOfObjects; ++i)
		objects.push_back(readObject(fin));

	// states' detail
	for (i=0; i<numOfState; ++i)
		states.push_back(readState(fin));

	return ObservedModel(num, objects, states);
}

/*
	Function: readInternalModel()
	Desc.	: read internal model info (imitation) through a stream interface and return an InternalModel
	Para.	: fin, a stream interface
	Return	: an InternalModel
*/
InternalModel readInternalModel (fstream &fin)
{
	int i, level, chosen, numOfObjects, numOfNodes;
	double reward, hCost;

	vector<Object> objects;

	// internal state
	InternalState intState;
	
	// policy and A* tree for one imitation
	vector<Node> policy, aStarTree;
	
	// read reward and number of node
	fin >> reward;
	fin >> numOfNodes;
	
	fin >> numOfObjects;
	// read internal objects which maybe different from demonstration
	for (i=0; i<numOfObjects; ++i)
		objects.push_back(readObject(fin));

	// read node in the A* tree
	for (i=0; i<numOfNodes; ++i)
	{
		// whether the node has been chosen during A* search
		fin >> chosen;
		fin >> level;
		fin >> hCost;

		// split it into policy and A* tree (only contain sibling)
		intState = readInternalState(fin);
		if (chosen)
			// hCost won't change, g and f will be calculated later
			policy.push_back(Node(intState, level, 0, hCost));
		else
			aStarTree.push_back(Node(intState, level, 0, hCost));
	}
	
	return InternalModel(objects, policy, aStarTree, reward);
}

/*
	Function: readAction()
	Desc.	: read action info through a stream interface and return an Action
	Para.	: fin, a stream interface
	Return	: an Action
*/
Action readAction(fstream &fin)
{
	int num;
	string name;
	double cost;

	State preConds, postConds;
	
	fin >> num;
	fin >> name;
	fin >> cost;

	// construct precondition
	preConds = readState(fin);
		
	// construct postcondition
	postConds = readState(fin);
	
	return Action(num, name, preConds, postConds, cost);
}

/*
	Function: isGoalState
	Desc.	: check whether the given state is goal state
	Para.	: intState, an internal state
	Return	: true if it is, otherwise false
*/
bool Imitation::isGoalState(const InternalState& intState)
{
	// check whether the given state corresponds to the last observed state
	return (intState.extStateNum ==currObservedStates.size()-1);
}

/*
	Function: calcHeuristicCost()
	Desc.	: calculate heuristic cost for the given state
	Para.	: intState, an internal state
	Return	: the distance between current observed state and final observed state
*/
double Imitation::calcHeuristicCost(const InternalState& intState)
{
	int goalState = currObservedStates.size()-1;

	double cost=0;
	for (int i=intState.extStateNum; i<goalState; ++i)	
		cost += (goalState-i)*COST_ATTRIBUTE;

	return cost;
}
/*
	Function: handCode
	Desc.	: a special version of calculating the distance between observed state and internal state
	Para.	: 
	Return	: double
*/
double Imitation::handCode(State& extState, State& intState)
{
	double hCost;

	vector<vector<string> > intStateString = stateToString(intState);
	vector<vector<string> > extStateString = stateToString(extState, false);
	
	// state difference
	double dist = extStateString.size();

	for (int i=0; i<extStateString.size(); ++i)
		dist -= similar(extStateString[i], intStateString[i]);

	if (extState == currObservedStates[currObservedStates.size()-1])
		hCost = dist * 200;
	else
		hCost = dist * 10;
	return hCost;
}
/*
	Function: stateToString()
	Desc	: replace the object with its attributes
	Para	: 
	Return	: string representation of state
*/
vector<vector<string> > Imitation::stateToString(State state, bool internal)
{
	int i;
	string actor;
	vector<Object> objs;
	
	vector<string> r;
	vector<vector<string> > s;

	vector<Relation>::const_iterator iter;
	vector<Object>::const_iterator objIter;

	if (internal)
	{
		actor = mMap[DEMO];
		objs = intObjects;	
	}
	else
	{
		actor = DEMO;
		objs = currObservedObjects;
	}
	
	for (i=0; i<objs.size(); ++i)
	{
		r.clear();
		// check whether there is a relationship between object and Demo/Imitator in the given state
		iter = state.findPattern(Relation("?", actor, objs[i].name));
		if (iter == state.end())
			// not exists, insert four EMPTY string (relation, Demo/Imitator and two attributes)
			r.insert(r.end(),4, "");
		else
		{
			r.push_back(iter->relation);
			r.push_back(iter->objA);

			// object attributes
			r.push_back(objs[i].color);
			r.push_back(objs[i].texture);
		}
		s.push_back(r);
	}

	 // There is special case for Gripper, check if an object on the gripper
	iter = state.findPattern(Relation("ON", "Gripper", "?"));
	r.clear();
	r.push_back("ON");
	r.push_back("Gripper");
	
	if (iter == state.end() || iter->objB == "NULL" || 
		(objIter = find_if(objs.begin(), objs.end(), sameName<Object>(iter->objB))) == objs.end())
		// nothing on gripper, two EMPTY string
		r.insert(r.end(),2, "");
	else
	{
		// find the object attributes
		r.push_back(objIter->color);
		r.push_back(objIter->texture);
	}
	s.push_back(r);

	return s;
}

double Imitation::similar(const vector<string>& extState, const vector<string>& intState)
{
	vector<Object>::iterator objIter;

	// garbage cleaning: check whether the trashcan exists, BLACK PLASTIC
	if (find(intObjects.begin(), intObjects.end(), Object("?", "BLACK", "PLASTIC")) != intObjects.end() && extState[2] == "GREEN")
	{
		// ignore the texture for GREEN object		
		return (extState[0] == intState[0] && extState[2] == intState[2]);
	}

	// Toy Collection: check whether the ToyCorner exists, BROWN WOOD, need consider different capabilities
	if (find(intObjects.begin(), intObjects.end(), Object("?", "BROWN", "WOOD"))!= intObjects.end())
	{
		if (extState[3] == "PLASTIC")
			if (extState[0] == "NEXT" && intState[0] == "")
				return 0.6;
			else
				// ignore the color for PLASTIC object				
				return (extState[0] == intState[0] && extState[3] == intState[3]);

		/*if (extState[2] == "BROWN" && extState[3] == "WOOD" && extState[2] == intState[2] && extState[3] == intState[3] && 
			extState[0] == "NEXT" && intState[0] == "AWAY")
			return 0.5;*/

		if (extState[0] == "ON" && extState[3] == "" && intState[3] == "PLASTIC")
			return 0.6;

		return (extState[0] == intState[0] && extState[2] == intState[2] && extState[3] == intState[3]);		
	}

	// futon match 1, check whether there are color match between object 0 and 2, if yes ignore the texture
	if (intObjects[0].color == intObjects[2].color)
	{
		//cout << "match with futon match"<< endl;		
		return (extState[0] == intState[0] && extState[2] == intState[2]);
	}
	
	return (extState[0] == intState[0] && extState[2]== intState[2] && extState[3] == intState[3]);
}

/*
	Function: simpleDistance
	Desc.	: a simple form of distance function
	Para.	: 
	Return	: double
*/
double Imitation::simpleDistance(const State& extState, const State& intState)
{
	double dist = 0;

	// represent the state with color/texture form in a fix order
	vector<vector<string> > intSState = stateToString(intState);
	vector<vector<string> > extSState = stateToString(extState,false);
	
	for (int i=0; i<extSState.size(); ++i)
		for (int j=0; j<extSState[i].size(); ++j)
			if (extSState[i][j] != intSState[i][j])
				++dist;

	//cout << "external state:" << endl;
	//for (int i=0; i<extSState.size(); ++i)
	//{
	//	for (int j=0; j<extSState[i].size(); ++j)
	//		cout << extSState[i][j] << " ";
	//	cout << endl;
	//}
	// 
	//cout << "internal state:" << endl;
	//for (int i=0; i<intSState.size(); ++i)
	//{
	//	for (int j=0; j<intSState[i].size(); ++j)
	//		cout << intSState[i][j] << " ";
	//	cout << endl;
	//}
	//cout << "distance: " << dist << endl << endl;
	return dist*10;
}
void Imitation::loadImitObjects (const string fileName)
{
	int i, j, numOfTasks, numOfObjects;
	vector<Object> objects;

	fstream fin;	
	fin.open(fileName.c_str(), ios::in);

	fin >> numOfTasks;
	for (i=0; i<numOfTasks; ++i)
	{
		objects.clear();

		fin >> numOfObjects;
		for (j=0; j<numOfObjects; ++j)
			objects.push_back(readObject(fin));

		imitObjects.push_back(objects);
	}
	fin.close();
}