#ifndef Bboxes_H
#define Bboxes_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream> //debug, not used in reality
#include "SequenceSample.h"
#include "testHOGdiff.h"

void findBoxes(SequenceSample& sample, std::vector<cv::Point2f>& centroids, float& H);


#endif
