#ifndef MODEL_EVALUATOR_H
#define MODEL_EVALUATOR_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <algorithm>

#include "action_utils.h"
#include "testHOGdiff.h"
#include "SequenceSample.h"

void DeploySVM(std::vector<SequenceSample>& samples);

#endif
