#ifndef FeatureExtractor_H
#define FeatureExtractor_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "SequenceSample.h"
#include <numeric>
#include <cmath>

float computeStrideCadence(const std::vector<float>& signal);
void extractFeatures (SequenceSample& sample, std::vector<cv::Point2f>& centroids, float H);

#endif