//Vlad Andries
#include "Model.h"

namespace fs = std::filesystem;

void DeploySVM(std::vector<SequenceSample>& samples) {
    int numFeats = static_cast<int>(samples[0].features.size());

    //train with leave-one-out cross-validation
    for (int i = 0; i < samples.size(); ++i) {
        std::vector<float> mean(numFeats, 0.0f);
        std::vector<float> stddev(numFeats, 0.0f);

        // Compute mean for each feature (except the i-th sample)
        for (int j = 0; j < samples.size(); ++j) {
            if (i == j) continue;

            for (int f = 0; f < numFeats; ++f)
                mean[f] += samples[j].features[f];
        }

        for (int f = 0; f < numFeats; ++f)
            mean[f] /= (samples.size() - 1);

        // Compute stddev for each feature (again, except the i-th sample)
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

        //Initialize training data and labels
        cv::Mat trainData(samples.size() - 1, numFeats, CV_32F);
        cv::Mat trainLabels(samples.size() - 1, 1, CV_32S);

        int trainIdx = 0;

        for (int j = 0; j < samples.size(); ++j) {
            if (i == j) continue;

            for (int f = 0; f < numFeats; ++f) {
                trainData.at<float>(trainIdx, f) = (samples[j].features[f] - mean[f]) / stddev[f]; //fill trainData with normalized features
            }

            trainLabels.at<int>(trainIdx, 0) = samples[j].trueLabel; //fill trainLabels with true labels
            trainIdx++;
        }
        
        //initialize SVM
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF); //setup a RBF Kernel 
        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 2500, 1e-6)); //max iterations and tolerance for convergence

        svm->trainAuto(trainData, cv::ml::ROW_SAMPLE, trainLabels, 10); //train svm with auto parameter selection using cross-validation
        cv::Mat testSample(1, numFeats, CV_32F);

        for (int f = 0; f < numFeats; ++f) {
            testSample.at<float>(0, f) = (samples[i].features[f] - mean[f]) / stddev[f]; //normalize the features of the test sample
        }

        int pred = static_cast<int>(svm->predict(testSample)); //svm label prediction for the test sample
        samples[i].predictedLabel = pred;
    }
}
