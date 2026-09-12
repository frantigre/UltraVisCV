#include "testHOGdiff.h"

cv::Mat difference(cv::Mat prev,cv::Mat succ){
    cv::Mat grayPrev,graySucc,diff;
    //cv::Mat mask; //remove very low differences, might be needed
    cv::cvtColor(prev,grayPrev,cv::COLOR_BGR2GRAY);
    cv::cvtColor(succ,graySucc,cv::COLOR_BGR2GRAY);
    absdiff(grayPrev,graySucc,diff);
    //eventually use mask for threshold and return mask
    return diff;
}

std::vector<cv::Mat> videoDiff(std::vector<cv::Mat> video){
    std::vector<cv::Mat> differences;
    for(size_t i=1; i<video.size();i++){
        cv::Mat diff = difference(video[i-1],video[i]);
        differences.push_back(diff);
    }
    return differences;
}

cv::HOGDescriptor generateHOG(cv::Mat frame){
    cv::HOGDescriptor hog(
        cv::Size(frame.cols, frame.rows),
        cv::Size(16, 16),
        cv::Size(8, 8),
        cv::Size(8, 8),
        9
    );
    return hog;
}

std::vector<float> computeHOG(cv::Mat img, cv::HOGDescriptor hog){
    std::vector<float> descriptor;
    hog.compute(img, descriptor);
    return descriptor;
}

std::vector<float> HOGVideo(std::vector<cv::Mat> videoDiff, cv::HOGDescriptor hog){
    std::vector<float> HOGVideo;
    for(size_t i=0; i<videoDiff.size();i++){
        std::vector<float> descriptor = computeHOG(videoDiff[i], hog);
        for(size_t j=0; j<descriptor.size(); j++){
            HOGVideo.push_back(descriptor[j]);
        }
    }
    return HOGVideo;
}

void shuffleData(cv::Mat& data, cv::Mat& cl, unsigned int seed){
    cv::Mat dataS = data.clone();
    cv::Mat clS = cl.clone(); //cl = classes
    std::vector<int> indexes;
    for(int i=0; i<data.rows;i++){
        indexes.push_back(i);
    }
    std::mt19937 generator(seed);
    std::shuffle(indexes.begin(), indexes.end(), generator);
    for(int i=0; i<data.rows; i++){
        data.row(indexes[i]).copyTo(dataS.row(i));
        cl.row(indexes[i]).copyTo(clS.row(i));
    }
    data=dataS;
    cl=clS;
}
