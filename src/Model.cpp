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


namespace fs = std::filesystem;

int numFeats = static_cast<int>(samples[0].features.size());
int correct = 0;
double totalIoU = 0.0;
int confusionMatrix[7][7] = {0};

ModelResult DeploySVM(std::vector<SequenceSample> samples){
    int N = static_cast<int>(samples.size());
    // Leave-One-Out Cross-Validation (LOOCV)
    for (int i = 0; i < N; ++i) {
        std::vector<float> mean(numFeats, 0.0f);
        std::vector<float> stddev(numFeats, 0.0f);

        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            for (int f = 0; f < numFeats; ++f) mean[f] += samples[j].features[f];
        }
        for (int f = 0; f < numFeats; ++f) mean[f] /= (N - 1);

        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            for (int f = 0; f < numFeats; ++f) {
                float diff = samples[j].features[f] - mean[f];
                stddev[f] += diff * diff;
            }
        }
        for (int f = 0; f < numFeats; ++f) {
            stddev[f] = std::sqrt(stddev[f] / (N - 1));
            if (stddev[f] < 1e-6f) stddev[f] = 1.0f;
        }

        cv::Mat trainData(N - 1, numFeats, CV_32F);
        cv::Mat trainLabels(N - 1, 1, CV_32S);

        int trainIdx = 0;
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            for (int f = 0; f < numFeats; ++f) {
                trainData.at<float>(trainIdx, f) = (samples[j].features[f] - mean[f]) / stddev[f];
            }
            trainLabels.at<int>(trainIdx, 0) = samples[j].trueLabel;
            trainIdx++;
        }

        // Automatic RBF SVM Cross-Validation Grid Search 
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 2500, 1e-6));

        svm->trainAuto(trainData, cv::ml::ROW_SAMPLE, trainLabels, 5);

        cv::Mat testSample(1, numFeats, CV_32F);
        for (int f = 0; f < numFeats; ++f) {
            testSample.at<float>(0, f) = (samples[i].features[f] - mean[f]) / stddev[f];
        }

        int pred = static_cast<int>(svm->predict(testSample));

        // Macro-Kinetic Physical Guardrails using Net Endpoint Drift
        float netTraverse = samples[i].netTranslationX;
        bool isStationaryPrediction = (pred == BOXING || pred == HANDWAVING || pred == HANDCLAPPING);
        bool isLocomotionPrediction = (pred == WALKING || pred == JOGGING || pred == RUNNING);

        // Guardrail 1: Moving actors cannot be classified as stationary
        /*if (netTraverse > 0.35f && isStationaryPrediction) {
            pred = WALKING;
        }
        // Guardrail 2: Stationary actors cannot be classified as locomotion
        if (netTraverse < 0.12f && isLocomotionPrediction) {
            // Re-route back to dominant stationary prediction
            pred = (samples[i].features[7] > 0.02f) ? HANDWAVING : HANDCLAPPING;
        }   
        
        */

        samples[i].predictedLabel = pred;
        int trueLabel = samples[i].trueLabel;

        confusionMatrix[trueLabel][samples[i].predictedLabel]++;
        if (samples[i].predictedLabel == trueLabel) correct++;
        totalIoU += samples[i].iou;

        saveAnnotatedFrame20(samples[i], outImgDir);

        std::cout << "Seq: " << samples[i].seqName
                  << " | True: " << std::left << std::setw(12) << getActionName(trueLabel)
                  << " | Pred: " << std::left << std::setw(12) << getActionName(samples[i].predictedLabel)
                  << " | IoU (Frame 20): " << std::fixed << std::setprecision(4) << samples[i].iou << "\n";
    }
    std::cout << "\n================ EVALUATION METRICS ================\n";
    std::cout << "Total Processed: " << N << " sequences\n";
    std::cout << "Global Accuracy: " << std::fixed << std::setprecision(2) << (static_cast<double>(correct) / N) * 100.0 << "%\n";
    std::cout << "Mean IoU (mIoU): " << std::fixed << std::setprecision(4) << (totalIoU / N) << "\n\n";

    std::cout << "Confusion Matrix (Rows: Ground Truth, Cols: Predicted):\n";
    std::cout << "\tWALK\tJOG\tRUN\tBOX\tWAVE\tCLAP\n";

    struct ModelRes res = {};
    res.correct = correct;
    res.totalIoU = totalIoU;
    res.confusionMatrix = confusionMatrix;
    res.N = N;

    return res;
}

void EvaluateModel(){
    for (int r = 1; r <= 6; ++r) {
        std::cout << getActionName(r).substr(0, 4) << "\t";
        for (int c = 1; c <= 6; ++c) {
            std::cout << confusionMatrix[r][c] << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "\nPer-Class Detailed Performance:\n";
    std::cout << std::left << std::setw(14) << "Class" 
              << std::setw(12) << "Precision" 
              << std::setw(12) << "Recall" 
              << std::setw(12) << "F1-Score" << "\n";
    std::cout << "--------------------------------------------------\n";

    double macroF1 = 0.0;
    for (int c = 1; c <= 6; ++c) {
        int tp = confusionMatrix[c][c];
        int fn = 0;
        int fp = 0;

        for (int j = 1; j <= 6; ++j) {
            if (j != c) {
                fn += confusionMatrix[c][j];
                fp += confusionMatrix[j][c];
            }
        }

        double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall    = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1        = (precision + recall > 1e-6) ? 2.0 * (precision * recall) / (precision + recall) : 0.0;
        macroF1 += f1;

        std::cout << std::left << std::setw(14) << getActionName(c)
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << precision
                  << std::setw(12) << recall
                  << std::setw(12) << f1 << "\n";
    }
    std::cout << "--------------------------------------------------\n";
    std::cout << "Macro Average F1-Score: " << std::fixed << std::setprecision(4) << (macroF1 / 6.0) << "\n";

    std::cout << "\nSaved " << N << " annotated frame-20 images to: " << fs::absolute(outImgDir).string() << "\n";

    std::cout.rdbuf(origCoutBuf);
}