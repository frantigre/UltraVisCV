#include "FileManagement.h"


void getDataset(std::vector<std::vector<cv::Mat>>& data, std::vector<GroundTruth>& labels){
    std::string root = "../UltraVisCV/dataset/Sequences";
    std::vector<std::string> types = {"boxing", "handclapping", "handwaving", "running", "jogging", "walking" };
    for(size_t i=0; i<types.size(); i++){
        std::string typePath = root+"/"+types[i];
        for (std::filesystem::directory_entry sample : std::filesystem::directory_iterator(typePath)){
            std::cout<<sample<<std::endl;
            if(sample.is_directory()){
                std::vector<cv::Mat> sampleData;
                GroundTruth sampleLabels;
                getSequence(sample.path().string(), sampleData, sampleLabels);
                data.push_back(sampleData);
                labels.push_back(sampleLabels);
            }
        }
    }
}

void getSequence (std::string path, std::vector<cv::Mat>& data, GroundTruth& labels){
    //std::cout<<"extracting from "<<path<<std::endl;;
    std::vector<std::string> files;
    cv::glob(path+"/data/*.png",files);
    for(size_t i=0; i<files.size(); i++){
        cv::Mat temporary = cv::imread(files[i]);
        data.push_back(temporary);
    }
    std::vector<float> temp={-1.0,-1.0,-1.0,-1.0,-1.0}; //initialization
    std::ifstream file(path+"/labels/ground_truth.txt");
    if(!file) throw std::runtime_error("nodata GetLabel");
    for(int i=0; i<5;i++){ //there are 5 data in the label file: type, xmin,ymin,xmax,ymax
        file>>temp[i];
    }
    labels = GroundTruth(temp[0],temp[1],temp[2],temp[3],temp[4]);
}
