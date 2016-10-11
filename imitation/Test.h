#include <string>

using namespace std;

class Test
{
	string a;
 public:
	Test(string s= ""):a(s){};
	~Test(void){};
	string toString(){return a;};
};

