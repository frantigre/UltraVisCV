#include <vector>
#include <opencv2/core.hpp>
#include "DataStructures.h"

//probably not needed float mIoU(std::vector<float>);

cv::Mat ConfMatrix(std::vector<GTruth> real, std::vector<int> predict); //can be string depending on how we classify

float F1Score(std::vector<GTruth> real, std::vector<int> predict, int picked_class); //can be string depending on how we classify


