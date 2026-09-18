#ifndef ACTION_UTILS_HPP
#define ACTION_UTILS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

#include "SequenceSample.h"
#include "FeatureExtractor.h"

/*enum ActionType {
    WALKING = 1,
    JOGGING = 2,
    RUNNING = 3,
    BOXING = 4,
    HANDWAVING = 5,
    HANDCLAPPING = 6
};*/ //sbagliato per non so quale motivo
enum ActionType{
    BOXING=1,
    HANDCLAPPING=2,
    HANDWAVING=3,
    JOGGING=4,
    RUNNING=5,
    WALKING=6
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
void saveAnnotatedFrame20(const SequenceSample& sample, std::filesystem::path outDir="outputImg");

cv::Mat difference(cv::Mat prev,cv::Mat succ);
std::vector<cv::Mat> videoDiff(std::vector<cv::Mat> video);
cv::Mat MeanOfDifferences(std::vector<cv::Mat> video);
double Gaussian(double x, double a, double b, double c, double d);

#endif // ACTION_UTILS_HPP
