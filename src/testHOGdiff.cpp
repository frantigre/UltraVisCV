#include "testHOGdiff.h"

cv::Mat difference(cv::Mat prev,cv::Mat succ){
    cv::Mat grayPrev,graySucc,diff;
    cv::Mat masked;  //removed very low diff
    //std::cout<<"prev ch "<<prev.channels()<<" succ ch "<<succ.channels()<<std::endl;
    if(prev.channels()==1) grayPrev=prev; //already grayscale
    else cv::cvtColor(prev,grayPrev,cv::COLOR_BGR2GRAY);
    if(succ.channels()==1) graySucc=succ; //already grayscale
    else cv::cvtColor(succ,graySucc,cv::COLOR_BGR2GRAY);
    absdiff(grayPrev,graySucc,diff);
    //might be good to choose the min value
    cv::threshold(diff,masked,10,255,cv::THRESH_TOZERO);
    return masked;
}

std::vector<cv::Mat> videoDiff(std::vector<cv::Mat> video){
    std::vector<cv::Mat> differences;
    for(size_t i=1; i<video.size();i++){
        cv::Mat diff = difference(video[i-1],video[i]);
        differences.push_back(diff);
    }
    return differences;
}

cv::Mat MeanOfDifferences(std::vector<cv::Mat> video) {
    std::vector<cv::Mat> diffs=videoDiff(video); //differences between every frame and next one
    cv::Mat sumDiff = cv::Mat::zeros(diffs[0].size(), CV_16U); //sum of all differences 16U since it will go beyond 255
    for(size_t i=0; i<diffs.size(); i++){
        cv::Mat tmp;
        diffs[i].convertTo(tmp, CV_16U);
        sumDiff+=diffs[i];
    }
    sumDiff/=static_cast<float>(diffs.size()); //brings back values to 0-255
    cv::Mat res;
    sumDiff.convertTo(res, CV_8U); //go back in 8 bit format
    return res;
}

double Gaussian(double x, double a, double b, double c, double d){
    double t;
    if (a>b || b>c || c>d) throw std::invalid_argument("Invalid values for generating Plateau");
    if (x>=b && x<=c) return 1;
    if (x<=a || x>=d) return 0;
    if (x>a && x<b) t=(b-x)/(b-a);
    if (x>c && x<d) t=(x-c)/(d-c);
    return std::exp(-3*(std::pow(t,3)));
}
