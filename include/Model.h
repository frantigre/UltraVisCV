#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "SequenceSample.h"
#include "FeatureExtractor.h"
#include "testHOGdiff.h"
#include "action_utils.h"


namespace fs = std::filesystem;

struct ModelResult
{
    int correct,
    int N,
    double totalIoU,
    int confusionMatrix[7][7]
};

void DeploySVM(std::vector<SequenceSample> samples);