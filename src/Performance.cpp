#include "Performance.h"

cv::Mat ConfMatrix(std::vector<GTruth> real, std::vector<int> predict){
    cv::Mat matrix = cv::Mat::zeros(6,6,CV_32S); //6 classes by definition
    for(size_t i=0; i<real.size(); i++){
        matrix.at<int>(real[i].label,predict[i])++;  //they need to be 0-5 so could be needed a -1
    }
    return matrix;
}

float F1Score(std::vector<GTruth> real, std::vector<int> predict, int picked_class){
    cv::Mat matrix = ConfMatrix(real, predict);
    int TP=matrix.at<int>(picked_class,picked_class);
    if(TP==0) return 0.0f; //no true positive, no need to compute anything + avoid division by 0
    int FP,FN; //TN not needed for F1
    for(int i=0; i<6;i++){ //6 classes by definition
        if(i!=picked_class){
        FP+=matrix.at<int>(i,picked_class);
        FN+=matrix.at<int>(picked_class,i);
        }
    }
    float precision=static_cast<float>(TP)/(TP+FP);
    float recall=static_cast<float>(TP)/(TP+FN);
    return 2*precision*recall/(precision+recall);
}

