#ifndef SequenceSample_H
#define SequenceSample_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>

class SequenceSample {
    public:
        std::string seqName;
        std::string frame20Path;
        cv::Rect bbox20;
        std::vector<float> features;
        float netTranslationX = 0.0f;
        int trueLabel = -1;
        int predictedLabel = -1;
        double iou = 0.0;
};

#endif