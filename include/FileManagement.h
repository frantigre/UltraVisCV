#ifndef FileManagement
#define FileManagement
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include <filesystem>
#include "DataStructures.h"

void getDataset (std::vector<std::vector<cv::Mat>>& data, std::vector<GTruth>& labels);
void getSequence (std::string path, std::vector<cv::Mat>& data, GTruth& labels);

#endif
