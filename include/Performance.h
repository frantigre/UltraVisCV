#include <vector>
#include <opencv2/core.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include "action_utils.h"


//probably not needed float mIoU(std::vector<float>);

cv::Mat ConfMatrix(std::vector<SequenceSample> samples); //can be string depending on how we classify

//float F1Score(std::vector<SequenceSample> samples, int picked_class); //can be string depending on how we classify

float F1Score(cv::Mat matrix, int picked_class);

void EvaluateModel(std::vector<SequenceSample> samples);

std::string centerText(const std::string& text, int width); 
