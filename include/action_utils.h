#ifndef ACTION_UTILS_HPP
#define ACTION_UTILS_HPP

#include <string>
#include <opencv2/core.hpp>

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
};

double computeIoU(const cv::Rect& a, const cv::Rect& b);
bool loadGroundTruth(const std::string& path, GroundTruth& gt);

#endif // ACTION_UTILS_HPP