//Vlad Andries
#include "action_utils.h"

namespace fs = std::filesystem;

std::string getActionName(int id) {
    //convert action ID to string
    switch(id) {
        case WALKING: return "walking";
        case JOGGING: return "jogging";
        case RUNNING: return "running";
        case BOXING: return "boxing";
        case HANDWAVING: return "waving";
        case HANDCLAPPING: return "clapping";
        default: return "unknown";
    }
}

int getActionIdFromName(const std::string& name) {
    std::string s = name;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    //convert action string to ID
    if (s.find("jogging") != std::string::npos) return JOGGING;
    if (s.find("running") != std::string::npos) return RUNNING;
    if (s.find("walking") != std::string::npos) return WALKING;
    if (s.find("boxing") != std::string::npos) return BOXING;
    if (s.find("waving") != std::string::npos || s.find("handwaving") != std::string::npos) return HANDWAVING;
    if (s.find("clapping") != std::string::npos || s.find("handclapping") != std::string::npos) return HANDCLAPPING;
    return -1;
}

GroundTruth::GroundTruth()=default;
GroundTruth::GroundTruth(float l, float xCent, float yCent, float width, float height){
    //storing action ID as integer and bbox as an opencv Rect2d object
    class_id=l;
    bbox = cv::Rect(xCent-width/2,yCent-height/2,width,height);
}


double computeIoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1); //intersection is the area of overlap
    int unionArea = a.area() + b.area() - interArea; //union is the sum of the two areas - the overlap

    return (unionArea <= 0) ? 0.0 : static_cast<double>(interArea) / unionArea; //return intersection over area
}


std::vector<cv::Mat> toGray(std::vector<cv::Mat> video){

    std::vector<cv::Mat> grays(video.size());
    for (size_t i=0; i<video.size(); ++i) {
        cv::cvtColor(video[i], grays[i], cv::COLOR_BGR2GRAY); // Convert to grayscale
    }
    return grays;
}

void saveAnnotatedFrame20(const SequenceSample& sample,std::filesystem::path outDir) {
    if (!std::filesystem::exists(outDir)) {
        std::filesystem::create_directories(outDir);
    }
    
    //revert img from gray to color
    cv::Mat frame20;
    cv::cvtColor(sample.frames[19], frame20, cv::COLOR_GRAY2BGR);

    std::string labelText = getActionName(sample.predictedLabel);
    //std::cout<<"text: "<<labelText;

    if(labelText=="unknown"){ //only draw the bbox for unknown actions
        cv::rectangle(frame20, sample.bboxes[19], cv::Scalar(0, 0, 255), 2);
        std::filesystem::path outFile= outDir/(sample.seqName + "_frame20.png");
        cv::imwrite(outFile.string(), frame20);
        //std::cout<<" no text"<<std::endl;
    }

    else{
        cv::rectangle(frame20, sample.bboxes[19], cv::Scalar(0, 0, 255), 2);
        cv::Point textPos;

        if (sample.bboxes[19].x > 70)
            textPos=cv::Point(std::max(7, sample.bboxes[19].x-55), std::max(20, sample.bboxes[19].y+15)); //place text to the left of the bbox if there's enough space
        else
            textPos=cv::Point(std::min(frame20.cols-60, sample.bboxes[19].x+sample.bboxes[19].width+5), std::max(20, sample.bboxes[19].y+15)); //else place it to its right

        cv::putText(frame20, labelText, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(0, 0, 255), 2); //and finally draw the label in red
        std::filesystem::path outFile=outDir/(sample.seqName + "_frame20.png");
        cv::imwrite(outFile.string(), frame20);
    }
}

cv::Mat difference(cv::Mat prev,cv::Mat succ){
    cv::Mat grayPrev,graySucc,diff;
    cv::Mat masked;  //removed very low diff

    if(prev.channels()==1) grayPrev=prev; //already grayscale
    else cv::cvtColor(prev,grayPrev,cv::COLOR_BGR2GRAY); //convert to grayscale
    if(succ.channels()==1) graySucc=succ; //already grayscale
    else cv::cvtColor(succ,graySucc,cv::COLOR_BGR2GRAY); //convert to grayscale
    
    absdiff(grayPrev,graySucc,diff); //compute the absolute difference between corresponding pixel values of the two greyscale frames
    
    //might be good to choose the min value
    cv::threshold(diff,masked,10,255,cv::THRESH_TOZERO);
    return masked;
}

std::vector<cv::Mat> videoDiff(std::vector<cv::Mat> video){ //computes the difference between each consecutive frames in the video and returns a vector of differences
    std::vector<cv::Mat> differences;

    for(size_t i=1; i<video.size();i++){
        cv::Mat diff = difference(video[i-1],video[i]); //call the difference function to compute the difference between the current frame and the previous one
        differences.push_back(diff); //store the difference in the vector
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
    
    if (x>=b && x<=c) return 1; //perfectly centered in the plateau, return 1
    if (x<=a || x>=d) return 0; //perfectly outside the plateau, return 0
    
    if (x>a && x<b) t=(b-x)/(b-a); //left side of the plateau, t goes from 1 to 0
    if (x>c && x<d) t=(x-c)/(d-c); //right side of the plateau, t goes from 0 to 1
    
    return std::exp(-3*(std::pow(t,3))); //returns a smooth Gaussian-like weighting function, with a plateau in the middle
}