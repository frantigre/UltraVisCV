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
#include "Bboxes.h"
#include "FileManagement.h"
#include "Model.h"
#include "Performance.h"


namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::vector<std::vector<cv::Mat>> database;
    std::vector<std::vector<cv::Mat>> DBGray; //might not be needed
    std::vector<GroundTruth> gt;
    getDataset(database, gt);
    std::vector<SequenceSample> samples(database.size());
    
    for(size_t i=0; i<database.size(); i++){
        std::cout<<"computing video "<<i+1<<std::endl;
        //DBGray.push_back(toGray(database[i])); not necessary probably
        float H; // meadian box H for feature normalization
        std::vector<cv::Point2f> centroids; // centroids of bboxes

        samples[i].frames=toGray(database[i]);
        samples[i].trueLabel=gt[i].class_id;
        samples[i].seqName="video"+std::to_string(i);
        
        findBoxes(samples[i], centroids, H);       //finds bboxes in the video included the one in frame 20
        extractFeatures(samples[i], centroids, H); //extract features for classification
        
        samples[i].iou = computeIoU(samples[i].bboxes[19], gt[i].bbox);     
        /*std::cout<<"\nvideo "<<i+1<<"\nbox20 "<<sample.bbox20<<std::endl;
        std::cout<<"grounTruth "<<gt[i].bbox<<std::endl;
        std::cout<<"IoU video "<<i+1<<" : "<<sample.iou<<std::endl;*/
        
        //saveAnnotatedFrame20(samples[i],"prova"); //writes file visualizing bbox and category of frame 20
    }
    DeploySVM(samples);
    EvaluateModel(samples);
}
