#ifndef ACTION_UTILS_HPP
#define ACTION_UTILS_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <filesystem>
#include <iostream> //degug
#include "SequenceSample.h"

enum ActionType {
    WALKING = 1,
    JOGGING = 2,
    RUNNING = 3,
    BOXING = 4,
    HANDWAVING = 5,
    HANDCLAPPING = 6
};

std::string getActionName(int id);
int getActionIdFromName(const std::string& name);

struct GroundTruth {
    int class_id;
    cv::Rect2d bbox;
    
    GroundTruth();
    GroundTruth(float l, float xCent, float yCent, float width, float height);
};

double computeIoU(const cv::Rect& a, const cv::Rect& b);
bool loadGroundTruth(const std::string& path, GroundTruth& gt);
std::vector<cv::Mat> toGray(std::vector<cv::Mat> video);
void saveAnnotatedFrame20(SequenceSample sample, std::filesystem::path outDir="outputImg");

#endif // ACTION_UTILS_HPP
