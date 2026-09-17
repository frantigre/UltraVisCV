#ifndef testHOGdiff
#define testHOGdiff
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream> //for debugging

cv::Mat difference(cv::Mat prev,cv::Mat succ);

std::vector<cv::Mat> videoDiff(std::vector<cv::Mat> video);

cv::Mat MeanOfDifferences(std::vector<cv::Mat> video);

double Gaussian(double x, double a, double b, double c, double d);

#endif
