//Samuele Volpato
#ifndef FileManagement_H
#define FileManagement_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include <filesystem>
#include "action_utils.h"

// load all dataset
void getDataset (std::vector<std::vector<cv::Mat>>& data, std::vector<GroundTruth>& labels, std::vector<std::string>& sampleNames, std::string root = "../KTH_Extracted_Dataset/Sequences");
// load single sample
void getSequence (std::string path, std::vector<cv::Mat>& data, GroundTruth& labels);

#endif
