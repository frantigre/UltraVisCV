//Francesco Ariani
#include <vector>
#include <opencv2/core.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include "action_utils.h"


cv::Mat ConfMatrix(std::vector<SequenceSample> samples); 

float F1Score(cv::Mat matrix, int picked_class);

void EvaluateModel(std::vector<SequenceSample> samples);

std::string centerText(std::string text, int width); 
