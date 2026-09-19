#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

#include "SequenceSample.h"
#include "FeatureExtractor.h"
#include "action_utils.h"
#include "Bboxes.h"
#include "FileManagement.h"
#include "Model.h"
#include "Performance.h"


int main(int argc, char** argv) {
    std::vector<std::vector<cv::Mat>> database;
    std::vector<GroundTruth> gt;
    std::vector<std::string> names; //saves name of folders containing data
    getDataset(database, gt, names);
    std::vector<SequenceSample> samples(database.size());
    
    for(size_t i=0; i<database.size(); i++){
        samples[i].seqName=names[i];
        std::cout<<"computing video "<<names[i]<<std::endl;
        float H; // meadian box H for feature normalization
        std::vector<cv::Point2f> centroids; // centroids of bboxes

        samples[i].frames=toGray(database[i]);
        samples[i].trueLabel=gt[i].class_id;
        
        findBoxes(samples[i], centroids, H);       //finds bboxes in the video included the one in frame 20
        extractFeatures(samples[i], centroids, H); //extract features for classification
        
        samples[i].iou = computeIoU(samples[i].bboxes[19], gt[i].bbox);     
    }
    DeploySVM(samples);
    EvaluateModel(samples);
}
