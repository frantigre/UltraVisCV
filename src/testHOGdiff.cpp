#include "testHOGdiff.h"

cv::Mat difference(cv::Mat prev,cv::Mat succ){
    cv::Mat grayPrev,graySucc,diff;
    cv::Mat masked;  //removed very low diff
    //std::cout<<"prev ch "<<prev.channels()<<" succ ch "<<succ.channels()<<std::endl;
    if(prev.channels()==1) grayPrev=prev; //already grayscale
    else cv::cvtColor(prev,grayPrev,cv::COLOR_BGR2GRAY);
    if(succ.channels()==1) graySucc=succ; //already grayscale
    else cv::cvtColor(succ,graySucc,cv::COLOR_BGR2GRAY);
    absdiff(grayPrev,graySucc,diff);
    //might be good to choose the min value
    cv::threshold(diff,masked,10,255,cv::THRESH_TOZERO);
    return masked;
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

cv::Mat computeTemporalEnergy(std::vector<cv::Mat> video) {
    std::vector<cv::Mat> diffs=videoDiff(video); //differences between every frame and next one
    cv::Mat sumDiff = cv::Mat::zeros(diffs[0].size(), CV_16U); //sum of all differences 16U since it will go beyond 255
    for(size_t i=0; i<diffs.size(); i++){
        cv::Mat tmp;
        diffs[i].convertTo(tmp, CV_16U);
        sumDiff+=diffs[i];
    }
    sumDiff/=static_cast<float>(diffs.size()); //brings back values to 0-255
    cv::Mat res;
    sumDiff.convertTo(res, CV_8U); //go back in 8 bit format
    return res;
}
    //old funct 
    /*int rows = grays[0].rows;
    int cols = grays[0].cols;
    int n = static_cast<int>(grays.size());

    cv::Mat meanImg(rows, cols, CV_32F, cv::Scalar(0));
    for (const auto& g : grays) {
        cv::Mat f;
        g.convertTo(f, CV_32F);
        meanImg += f;
    }
    meanImg /= static_cast<float>(n);

    cv::Mat varImg(rows, cols, CV_32F, cv::Scalar(0));
    for (const auto& g : grays) {
        cv::Mat f, diff;
        g.convertTo(f, CV_32F);
        diff = f - meanImg;
        varImg += diff.mul(diff);
    }
    varImg /= static_cast<float>(n);

    cv::Mat stdImg;
    cv::sqrt(varImg, stdImg);
    cv::Mat energy8U;
    stdImg.convertTo(energy8U, CV_8U);
    return energy8U;
}*/

//std::vector<cv::Mat> toGray(std::vector<cv::Mat> video);

double Gaussian(double x, double a, double b, double c, double d){
    double t;
    if (a>b || b>c || c>d) throw std::invalid_argument("Invalid values for generating Plateau");
    if (x>=b && x<=c) return 1;
    if (x<=a || x>=d) return 0;
    if (x>a && x<b) t=(b-x)/(b-a);
    if (x>c && x<d) t=(x-c)/(d-c);
    return std::exp(-3*(std::pow(t,3)));
}
