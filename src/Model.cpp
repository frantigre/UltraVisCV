#include "Model.h"

namespace fs = std::filesystem;

/*
    TO DO:  
            std::vector<SequenceSample>& samples nel costruttore
            Dipendenze da   saveAnnotatedFrame20,
                            OutImgDir,
                            origCoutBuf
*/

void DeploySVM(std::vector<SequenceSample>& samples) {
    int numFeats = static_cast<int>(samples[0].features.size());
    int correct = 0;
    float totalIoU = 0.0;
    int confusionMatrix[7][7] = {0};

    // Leave-One-Out Cross-Validation (LOOCV)
    for (int i = 0; i < samples.size(); ++i) {
        std::vector<float> mean(numFeats, 0.0f);
        std::vector<float> stddev(numFeats, 0.0f);

        for (int j = 0; j < samples.size(); ++j) {
            if (i == j) continue;

            for (int f = 0; f < numFeats; ++f)
                mean[f] += samples[j].features[f];
        }

        for (int f = 0; f < numFeats; ++f)
            mean[f] /= (samples.size() - 1);

        for (int j = 0; j < samples.size(); ++j) {
            if (i == j) continue;

            for (int f = 0; f < numFeats; ++f) {
                float diff = samples[j].features[f] - mean[f];
                stddev[f] += diff * diff;
            }
        }

        for (int f = 0; f < numFeats; ++f) {
            stddev[f] = std::sqrt(stddev[f] / (samples.size() - 1));
            if (stddev[f] < 1e-6f) stddev[f] = 1.0f;
        }

        cv::Mat trainData(samples.size() - 1, numFeats, CV_32F);
        cv::Mat trainLabels(samples.size() - 1, 1, CV_32S);

        int trainIdx = 0;

        for (int j = 0; j < samples.size(); ++j) {
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

        samples[i].predictedLabel = pred;
        int trueLabel = samples[i].trueLabel;

        confusionMatrix[trueLabel][samples[i].predictedLabel]++;

        if (samples[i].predictedLabel == trueLabel)
            correct++;

        totalIoU += samples[i].iou;

        std::cout << "Seq: " << samples[i].seqName
                  << " | True: " << std::left << std::setw(12) << getActionName(trueLabel)
                  << " | Pred: " << std::left << std::setw(12) << getActionName(samples[i].predictedLabel)
                  << " | IoU (Frame 20): " << std::fixed << std::setprecision(4) << samples[i].iou << "\n";
    }
}
