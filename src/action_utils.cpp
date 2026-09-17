#include <iostream>
#include <string>

#include <fstream>
#include <algorithm>

#include "SequenceSample.h"
#include "FeatureExtractor.h"
#include "testHOGdiff.h"
#include "action_utils.h"

namespace fs = std::filesystem;

std::string getActionName(int id) {
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
    class_id=l;
    bbox = cv::Rect(xCent-width/2,yCent-height/2,width,height);
}


double computeIoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;

    return (unionArea <= 0) ? 0.0 : static_cast<double>(interArea) / unionArea;
}

bool loadGroundTruth(const std::string& path, GroundTruth& gt) {        //abbastanza sicuro non serva
    std::ifstream file(path);
    if (!file.is_open()) return false;
    file >> gt.class_id >> gt.bbox.x >> gt.bbox.y >> gt.bbox.width >> gt.bbox.height;
    return true;
}


std::vector<cv::Mat> toGray(std::vector<cv::Mat> video){

    std::vector<cv::Mat> grays(video.size());
    for (size_t i=0; i<video.size(); ++i) {
        cv::cvtColor(video[i], grays[i], cv::COLOR_BGR2GRAY);
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
    if(labelText=="unknown"){
        cv::rectangle(frame20, sample.bboxes[19], cv::Scalar(0, 0, 255), 2);
        std::filesystem::path outFile= outDir/(sample.seqName + "_frame20.png");
        cv::imwrite(outFile.string(), frame20);
        //std::cout<<" no text"<<std::endl;
    }
    else{

        cv::rectangle(frame20, sample.bboxes[19], cv::Scalar(0, 0, 255), 2);

        cv::Point textPos;
        if (sample.bboxes[19].x > 80) {
            textPos=cv::Point(std::max(10, sample.bboxes[19].x-75), std::max(20, sample.bboxes[19].y+15));
        } else {
            textPos=cv::Point(std::min(frame20.cols-80, sample.bboxes[19].x+sample.bboxes[19].width+5), std::max(20, sample.bboxes[19].y+15));
        }

        cv::putText(frame20, labelText, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 255), 2);
        std::filesystem::path outFile=outDir/(sample.seqName + "_frame20.png");
        cv::imwrite(outFile.string(), frame20);
    }
}
