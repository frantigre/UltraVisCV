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

double computeIoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;

    return (unionArea <= 0) ? 0.0 : static_cast<double>(interArea) / unionArea;
}

bool loadGroundTruth(const std::string& path, GroundTruth& gt) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    file >> gt.class_id >> gt.bbox.x >> gt.bbox.y >> gt.bbox.width >> gt.bbox.height;
    return true;
}