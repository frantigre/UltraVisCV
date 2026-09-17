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

/*class TeeBuffer : public std::streambuf {
public:
    TeeBuffer(std::streambuf* sb1, std::streambuf* sb2) : sb1_(sb1), sb2_(sb2) {}
protected:
    virtual int overflow(int c) override {
        if (c == EOF) return !EOF;
        int const r1 = sb1_->sputc(c);
        int const r2 = sb2_->sputc(c);
        return (r1 == EOF || r2 == EOF) ? EOF : c;
    }
    virtual int sync() override {
        int const r1 = sb1_->pubsync();
        int const r2 = sb2_->pubsync();
        return (r1 == 0 && r2 == 0) ? 0 : -1;
    }
private:
    std::streambuf* sb1_;
    std::streambuf* sb2_;
};*/

/*int main(int argc, char** argv) {
    std::ofstream outFile("output.txt");
    std::streambuf* origCoutBuf = std::cout.rdbuf();
    std::unique_ptr<TeeBuffer> tee;
    if (outFile.is_open()) {
        tee = std::make_unique<TeeBuffer>(origCoutBuf, outFile.rdbuf());
        std::cout.rdbuf(tee.get());
    }

    fs::path rootPath = (argc >= 2) ? argv[1] : "../UltraVisCV/dataset";
    if (!fs::exists(rootPath) || !fs::is_directory(rootPath)) {
        std::cerr << "Invalid dataset directory: " << rootPath << "\n";
        std::cout.rdbuf(origCoutBuf);
        return -1;
    }

    fs::path outImgDir = "outputImg";
    if (!fs::exists(outImgDir)) {
        fs::create_directories(outImgDir);
    }

    std::cout << "Dataset path: " << rootPath << "\n";
    std::cout << "Annotated Frame 20 Output: " << fs::absolute(outImgDir).string() << "\n\n";

    std::vector<fs::path> sequenceDirs;
    for (const auto& entry : fs::recursive_directory_iterator(rootPath)) {
        if (entry.is_directory() && entry.path().filename() == "data") {
            sequenceDirs.push_back(entry.path().parent_path());
        }
    }
    std::sort(sequenceDirs.begin(), sequenceDirs.end());

    std::cout << "Extracting features from " << sequenceDirs.size() << " sequences...\n";
    std::vector<SequenceSample> samples[i];
    for (const auto& seq : sequenceDirs) {
        SequenceSample s;
        if (extractSequenceFeatures(seq, s)) {
            samples[i].push_back(s);
        }
    }
    
    ModelEvaluator model;
    model.DeploySVM(samples[i]);
    model.EvaluateModel();

        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            for (int f = 0; f < numFeats; ++f) mean[f] += samples[i][j].features[f];
        }
        for (int f = 0; f < numFeats; ++f) mean[f] /= (N - 1);

        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            for (int f = 0; f < numFeats; ++f) {
                float diff = samples[i][j].features[f] - mean[f];
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
                trainData.at<float>(trainIdx, f) = (samples[i][j].features[f] - mean[f]) / stddev[f];
            }
            trainLabels.at<int>(trainIdx, 0) = samples[i][j].trueLabel;
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
            testSample.at<float>(0, f) = (samples[i][i].features[f] - mean[f]) / stddev[f];
        }

        int pred = static_cast<int>(svm->predict(testSample));

        // Macro-Kinetic Physical Guardrails using Net Endpoint Drift
        float netTraverse = samples[i][i].netTranslationX;
        bool isStationaryPrediction = (pred == BOXING || pred == HANDWAVING || pred == HANDCLAPPING);
        bool isLocomotionPrediction = (pred == WALKING || pred == JOGGING || pred == RUNNING);

        samples[i][i].predictedLabel = pred;
        int trueLabel = samples[i][i].trueLabel;

        confusionMatrix[trueLabel][samples[i][i].predictedLabel]++;
        if (samples[i][i].predictedLabel == trueLabel) correct++;
        totalIoU += samples[i][i].iou;

        saveAnnotatedFrame20(samples[i][i], outImgDir);

        std::cout << "Seq: " << samples[i][i].seqName
                  << " | True: " << std::left << std::setw(12) << getActionName(trueLabel)
                  << " | Pred: " << std::left << std::setw(12) << getActionName(samples[i][i].predictedLabel)
                  << " | IoU (Frame 20): " << std::fixed << std::setprecision(4) << samples[i][i].iou << "\n";
    }

    std::cout << "\n================ EVALUATION METRICS ================\n";
    std::cout << "Total Processed: " << N << " sequences\n";
    std::cout << "Global Accuracy: " << std::fixed << std::setprecision(2) << (static_cast<double>(correct) / N) * 100.0 << "%\n";
    std::cout << "Mean IoU (mIoU): " << std::fixed << std::setprecision(4) << (totalIoU / N) << "\n\n";

    std::cout << "Confusion Matrix (Rows: Ground Truth, Cols: Predicted):\n";
    std::cout << "\tWALK\tJOG\tRUN\tBOX\tWAVE\tCLAP\n";
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
    return 0;
}*/

int main(int argc, char** argv) {
    std::vector<std::vector<cv::Mat>> database;
    std::vector<std::vector<cv::Mat>> DBGray; //might not be needed
    std::vector<GroundTruth> gt;
    getDataset(database, gt);
    //std::cout<<"CARICA (credo)"<<std::endl;
    std::vector<SequenceSample> samples(database.size());
    
    for(size_t i=0; i<database.size(); i++){
        std::cout<<"computing video "<<i+1<<std::endl;
        //DBGray.push_back(toGray(database[i])); not necessary probably
        std::vector<cv::Mat> videoG=toGray(database[i]);
        std::vector<cv::Rect> bboxes;
        float H;
        
        samples[i].trueLabel=gt[i].class_id;
        samples[i].frame20=database[i][19]; //19 index = frame 20
        samples[i].seqName="video"+std::to_string(i);
        
        findBoxes(videoG, samples[i], bboxes, H);       //finds bboxes in the video included the one in frame 20
        extractFeatures(samples[i], videoG, bboxes, H); //extract features for classification
        
        samples[i].iou = computeIoU(samples[i].bbox20, gt[i].bbox);     
        /*std::cout<<"\nvideo "<<i+1<<"\nbox20 "<<sample.bbox20<<std::endl;
        std::cout<<"grounTruth "<<gt[i].bbox<<std::endl;
        std::cout<<"IoU video "<<i+1<<" : "<<sample.iou<<std::endl;*/
        
        //saveAnnotatedFrame20(samples[i],"prova"); //writes file visualizing bbox and category of frame 20
    }
    DeploySVM(samples);
    EvaluateModel(samples);
}
