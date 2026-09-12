#ifndef DataStructures
#define DataStructures
#include <algorithm>

class Box{

public:

    int xmin, ymin, xmax, ymax;

    Box(int x_min=-1, int y_min=-1, int x_max=-1, int y_max=-1);
    int GetArea();
    float IoU(Box box2);

};

class GTruth{
    public:
    
    Box box;
    int label;
    
    GTruth(int l=-1, int x_min=-1, int y_min=-1, int x_max=-1, int y_max=-1);
};

#endif
