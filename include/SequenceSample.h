#ifndef SequenceSample_H
#define SequenceSample_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>

class SequenceSample {
    public:
        std::string seqName;
        std::vector<cv::Mat> frames; // all gray frames of sample
        std::vector<cv::Rect> bboxes;
        std::vector<float> features;
        int trueLabel = -1;
        int predictedLabel = -1;
        double iou = 0.0;
};

#endif