#ifndef testHOGdiff
#define testHOGdiff
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <algorithm>
#include <random>

cv::Mat difference(cv::Mat prev,cv::Mat succ); //keep

std::vector<cv::Mat> videoDiff(std::vector<cv::Mat> video); //keep

cv::HOGDescriptor generateHOG(cv::Mat frame); //no good

std::vector<float> computeHOG(cv::Mat img, cv::HOGDescriptor hog); //no good

std::vector<float> HOGVideo(std::vector<cv::Mat> video, cv::HOGDescriptor hog); //no good

void shuffleData(cv::Mat& data, cv::Mat& classifiers, unsigned int seed=0); //keep

#endif
