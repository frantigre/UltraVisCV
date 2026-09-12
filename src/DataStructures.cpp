#include "DataStructures.h"


Box::Box(int x_min, int y_min, int x_max,int y_max){
    xmin = x_min;
    ymin = y_min;
    xmax = x_max;
    ymax = y_max;
}

int Box::GetArea(){
    return (xmax-xmin)*(ymax-ymin);
}

float Box::IoU(Box box2){
    Box intersection;
    if((this->xmin>box2.xmax)||(box2.xmin>this->xmax)||(this->ymin>box2.ymax)||(box2.ymin>this->ymax)) intersection = Box(-1,-1,-1,-1); //no intersection = error box
    else intersection = Box(std::max(this->xmin,box2.xmin),std::max(this->ymin,box2.ymin),std::min(this->xmax,box2.xmax),std::min(this->ymax,box2.ymax));
    return static_cast<float>(intersection.GetArea())/this->GetArea()+box2.GetArea()-intersection.GetArea();
}

GTruth::GTruth(int l, int x_min, int y_min, int x_max, int y_max){
    box = Box(x_min,y_min,x_max,y_max);
    label= l;
}


