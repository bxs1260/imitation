objects = Random.o NeuralNetwork.o FeedForward.o Object.o Relation.o Action.o State.o \
	InternalState.o InternalModel.o ObservedModel.o Imitation.o
 
imitation : $(objects)
	g++ -O -o imitation main.cpp $(objects)

Object.o : Object.cpp Object.h
	g++ -c -O Object.cpp
Relation.o : Relation.cpp Relation.h
	g++ -c -O Relation.cpp
Action.o : Action.cpp Action.h
	g++ -c -O Action.cpp
State.o : State.cpp State.h
	g++ -c -O State.cpp
InternalState.o : InternalState.cpp InternalState.h
	g++ -c -O InternalState.cpp
InternalModel.o : InternalModel.cpp InternalModel.h
	g++ -c -O InternalModel.cpp
ObservedModel.o : ObservedModel.cpp ObservedModel.h
	g++ -c -O ObservedModel.cpp
Imitation.o : Imitation.cpp Imitation.h tree.h
	g++ -c -O Imitation.cpp

Random.o : Random.cpp Random.h
	g++ -c -O Random.cpp
NeuralNetwork.o : NeuralNetwork.cpp NeuralNetwork.h
	g++ -c -O NeuralNetwork.cpp
FeedForward.o : FeedForward.cpp FeedForward.h
	g++ -c -O FeedForward.cpp
 
clean: 
	rm imitation $(objects)