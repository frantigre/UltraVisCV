#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

enum ActionType {
    WALKING = 1,
    JOGGING = 2,
    RUNNING = 3,
    BOXING = 4,
    HANDWAVING = 5,
    HANDCLAPPING = 6
};

std::string getActionName(int id) {
    switch(id) {
        case WALKING: return "walking";
        case JOGGING: return "jogging";
        case RUNNING: return "running";
        case BOXING: return "boxing";
        case HANDWAVING: return "handwaving";
        case HANDCLAPPING: return "handclapping";
        default: return "unknown";
    }
}

int getActionIdFromName(const std::string& name) {
    std::string s = name;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s.find("jogging") != std::string::npos) return JOGGING;
    if (s.find("running") != std::string::npos) return RUNNING;
    if (s.find("walking") != std::string::npos) return WALKING;
    if (s.find("boxing") != std::string::npos) return BOXING;
    if (s.find("waving") != std::string::npos || s.find("handwaving") != std::string::npos) return HANDWAVING;
    if (s.find("clapping") != std::string::npos || s.find("handclapping") != std::string::npos) return HANDCLAPPING;
    return -1;
}

struct GroundTruth {
    int class_id;
    cv::Rect2d bbox;
};

double computeIoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;

    return (unionArea <= 0) ? 0.0 : static_cast<double>(interArea) / unionArea;
}

bool loadGroundTruth(const std::string& path, GroundTruth& gt) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    file >> gt.class_id >> gt.bbox.x >> gt.bbox.y >> gt.bbox.width >> gt.bbox.height;
    return true;
}

struct SequenceSample {
    std::string seqName;
    std::vector<std::string> framePaths;
    std::vector<cv::Rect> bboxes;
    std::vector<float> features;
    int trueLabel = -1;
    int predictedLabel = -1;
    double iou = 0.0;
};

// Temporal Standard Deviation Image to extract stationary actors
cv::Mat computeTemporalEnergy(const std::vector<cv::Mat>& grays) {
    int rows = grays[0].rows;
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
}

bool extractSequenceFeatures(const fs::path& seqDir, SequenceSample& sample) {
    sample.seqName = seqDir.filename().string();
    sample.trueLabel = getActionIdFromName(sample.seqName);
    if (sample.trueLabel == -1) {
        sample.trueLabel = getActionIdFromName(seqDir.parent_path().filename().string());
    }
    if (sample.trueLabel == -1) return false;

    fs::path dataDir = seqDir / "data";
    fs::path labelsDir = seqDir / "labels";

    sample.framePaths.clear();
    if (fs::exists(dataDir) && fs::is_directory(dataDir)) {
        for (const auto& entry : fs::directory_iterator(dataDir)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".png" || ext == ".bmp" || ext == ".jpeg") {
                sample.framePaths.push_back(entry.path().string());
            }
        }
    }
    std::sort(sample.framePaths.begin(), sample.framePaths.end());
    if (sample.framePaths.size() < 20) return false;

    GroundTruth gt;
    bool hasGT = false;
    if (fs::exists(labelsDir) && fs::is_directory(labelsDir)) {
        for (const auto& entry : fs::directory_iterator(labelsDir)) {
            if (entry.path().extension() == ".txt") {
                hasGT = loadGroundTruth(entry.path().string(), gt);
                if (hasGT) break;
            }
        }
    }

    std::vector<cv::Mat> grays(sample.framePaths.size());
    for (size_t i = 0; i < sample.framePaths.size(); ++i) {
        cv::Mat bgr = cv::imread(sample.framePaths[i]);
        cv::cvtColor(bgr, grays[i], cv::COLOR_BGR2GRAY);
    }

    // 1. Compute dynamic motion energy field across entire clip
    cv::Mat energy = computeTemporalEnergy(grays);
    cv::Mat energyMask;
    cv::threshold(energy, energyMask, 12, 255, cv::THRESH_BINARY);
    cv::morphologyEx(energyMask, energyMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 25)));

    std::vector<std::vector<cv::Point>> eContours;
    cv::findContours(energyMask, eContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Rect globalActiveZone(0, 0, grays[0].cols, grays[0].rows);
    double maxEArea = 0;
    for (const auto& c : eContours) {
        double a = cv::contourArea(c);
        if (a > maxEArea && a > 400.0) {
            maxEArea = a;
            globalActiveZone = cv::boundingRect(c);
        }
    }

    // 2. Track dynamic frame-by-frame bounding boxes using three-frame differencing
    sample.bboxes.clear();
    std::vector<cv::Point2f> centroids;
    std::vector<float> aspectRatios;
    std::vector<float> actorHeights;

    for (size_t i = 0; i < grays.size(); ++i) {
        size_t prevIdx = (i > 0) ? (i - 1) : 0;
        size_t nextIdx = (i + 1 < grays.size()) ? (i + 1) : i;

        cv::Mat diff1, diff2, motionMask;
        cv::absdiff(grays[i], grays[prevIdx], diff1);
        cv::absdiff(grays[i], grays[nextIdx], diff2);
        cv::bitwise_and(diff1, diff2, motionMask);
        cv::threshold(motionMask, motionMask, 15, 255, cv::THRESH_BINARY);
        cv::morphologyEx(motionMask, motionMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 15)));

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(motionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Rect curBox(0, 0, 0, 0);
        double maxA = 0;
        for (const auto& c : contours) {
            double a = cv::contourArea(c);
            if (a > 150.0 && a > maxA) {
                maxA = a;
                curBox = cv::boundingRect(c);
            }
        }

        // Anchor stationary actors to full height if motion box collapses to hands
        if (curBox.area() > 0) {
            if (curBox.height < globalActiveZone.height * 0.55) {
                // Expanding height downward to capture stationary torso/legs
                curBox.y = std::max(0, globalActiveZone.y);
                curBox.height = std::min(grays[0].rows - curBox.y, globalActiveZone.height);
            }
            sample.bboxes.push_back(curBox);
            centroids.push_back(cv::Point2f(curBox.x + curBox.width / 2.0f, curBox.y + curBox.height / 2.0f));
            aspectRatios.push_back(static_cast<float>(curBox.width) / curBox.height);
            actorHeights.push_back(static_cast<float>(curBox.height));
        } else if (!sample.bboxes.empty()) {
            sample.bboxes.push_back(sample.bboxes.back());
            centroids.push_back(centroids.back());
            aspectRatios.push_back(aspectRatios.back());
            actorHeights.push_back(actorHeights.back());
        } else {
            sample.bboxes.push_back(globalActiveZone);
            centroids.push_back(cv::Point2f(globalActiveZone.x + globalActiveZone.width / 2.0f, globalActiveZone.y + globalActiveZone.height / 2.0f));
            aspectRatios.push_back(static_cast<float>(globalActiveZone.width) / globalActiveZone.height);
            actorHeights.push_back(static_cast<float>(globalActiveZone.height));
        }
    }

    // 3. Dense Optical Flow Analysis
    double totalVxUpper = 0.0;
    double totalVyUpper = 0.0;
    double totalConvergence = 0.0;
    int flowCount = 0;
    double totalUpperWidth = 0.0;

    for (size_t i = 1; i < grays.size(); ++i) {
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(grays[i - 1], grays[i], flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        cv::Rect curBBox = sample.bboxes[i];
        // Focus on upper 60% of bounding box (arms/head)
        cv::Rect upperROI(curBBox.x, curBBox.y, curBBox.width, static_cast<int>(curBBox.height * 0.60));
        upperROI &= cv::Rect(0, 0, flow.cols, flow.rows);

        if (upperROI.area() > 0) {
            totalUpperWidth += upperROI.width;
            float midX = upperROI.x + upperROI.width / 2.0f;

            for (int r = upperROI.y; r < upperROI.y + upperROI.height; ++r) {
                for (int c = upperROI.x; c < upperROI.x + upperROI.width; ++c) {
                    cv::Point2f f = flow.at<cv::Point2f>(r, c);
                    float mag = std::hypot(f.x, f.y);
                    if (mag > 0.6f) {
                        totalVxUpper += std::abs(f.x);
                        totalVyUpper += std::abs(f.y);
                        if (c < midX) totalConvergence += f.x;
                        else          totalConvergence -= f.x;
                        flowCount++;
                    }
                }
            }
        }
    }

    // Scale Normalization Factor: Median Actor Height
    std::sort(actorHeights.begin(), actorHeights.end());
    float H = std::max(20.0f, actorHeights[actorHeights.size() / 2]);

    // Displacement & Translational Speed
    float totalDisplacementX = 0.0f;
    float maxInstantVx = 0.0f;
    for (size_t i = 1; i < centroids.size(); ++i) {
        float instVx = std::abs(centroids[i].x - centroids[i - 1].x);
        if (instVx > maxInstantVx) maxInstantVx = instVx;
    }
    if (centroids.size() > 1) {
        totalDisplacementX = std::abs(centroids.back().x - centroids.front().x);
    }
    float normVelocityX = (totalDisplacementX / centroids.size()) / H;
    float normMaxInstantVx = maxInstantVx / H;

    // Aspect Ratio Variance (Stride Oscillation)
    float meanAR = std::accumulate(aspectRatios.begin(), aspectRatios.end(), 0.0f) / aspectRatios.size();
    float varAR = 0.0f;
    for (float ar : aspectRatios) varAR += (ar - meanAR) * (ar - meanAR);
    varAR /= aspectRatios.size();

    // Scale-Invariant Optical Flow Features
    float normVy = (flowCount > 0) ? static_cast<float>((totalVyUpper / flowCount) / H) : 0.0f;
    float normVx = (flowCount > 0) ? static_cast<float>((totalVxUpper / flowCount) / H) : 0.0f;
    float normConv = (flowCount > 0) ? static_cast<float>((totalConvergence / flowCount) / H) : 0.0f;
    float normUpperWidth = (grays.size() > 1) ? static_cast<float>((totalUpperWidth / (grays.size() - 1)) / H) : 0.0f;
    float flowRatio = (normVx + 1e-5f) / (normVy + 1e-5f);

    // 8 Scale-Normalized Features
    sample.features = {
        normVelocityX,
        normMaxInstantVx,
        varAR,
        normVy,
        normVx,
        normConv,
        normUpperWidth,
        flowRatio
    };

    // Calculate IoU on the designated median reference frame (20th frame, index 19)
    if (hasGT && grays.size() >= 20) {
        int W = grays[19].cols;
        int imgH = grays[19].rows;
        double gx_c = gt.bbox.x;
        double gy_c = gt.bbox.y;
        double gw = gt.bbox.width;
        double gh = gt.bbox.height;

        if (gx_c <= 1.0 && gy_c <= 1.0 && gw <= 1.0 && gh <= 1.0) {
            gx_c *= W; gy_c *= imgH; gw *= W; gh *= imgH;
        }
        int gx = static_cast<int>(gx_c - gw / 2.0);
        int gy = static_cast<int>(gy_c - gh / 2.0);
        cv::Rect gtBox(gx, gy, static_cast<int>(gw), static_cast<int>(gh));

        sample.iou = computeIoU(sample.bboxes[19], gtBox);
    } else {
        sample.iou = 0.0;
    }

    return true;
}

void renderSequenceVisual(const SequenceSample& sample) {
    std::string winName = "HRI Action Recognition Output";
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    cv::resizeWindow(winName, 640, 480);

    std::string labelText = getActionName(sample.predictedLabel);

    for (size_t i = 0; i < sample.framePaths.size(); ++i) {
        cv::Mat frame = cv::imread(sample.framePaths[i]);
        if (frame.empty()) continue;

        cv::Rect bbox = sample.bboxes[i];
        cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);

        cv::Point textPos;
        if (bbox.x > 80) {
            textPos = cv::Point(std::max(10, bbox.x - 75), std::max(20, bbox.y + 15));
        } else {
            textPos = cv::Point(std::min(frame.cols - 80, bbox.x + bbox.width + 5), std::max(20, bbox.y + 15));
        }

        cv::putText(frame, labelText, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 255), 2);
        cv::imshow(winName, frame);

        char key = static_cast<char>(cv::waitKey(40));
        if (key == 27 || key == 'q') {
            return;
        }
    }
}

class TeeBuffer : public std::streambuf {
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
};

int main(int argc, char** argv) {
    std::ofstream outFile("output.txt");
    std::streambuf* origCoutBuf = std::cout.rdbuf();
    std::unique_ptr<TeeBuffer> tee;
    if (outFile.is_open()) {
        tee = std::make_unique<TeeBuffer>(origCoutBuf, outFile.rdbuf());
        std::cout.rdbuf(tee.get());
    }

    const cv::String keys =
        "{help h usage ? |          | Print this help message }"
        "{@dataset       |../dataset| Path to the root dataset folder }"
        "{visual v       |          | Enable visual output playback (bounding box + labels) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("KTH Action Recognition for HRI");

    if (parser.has("help")) {
        parser.printMessage();
        std::cout.rdbuf(origCoutBuf);
        return 0;
    }

    fs::path rootPath = parser.get<std::string>("@dataset");
    bool showVisual = parser.has("visual");

    if (!parser.check()) {
        parser.printErrors();
        std::cout.rdbuf(origCoutBuf);
        return -1;
    }

    if (!fs::exists(rootPath) || !fs::is_directory(rootPath)) {
        std::cerr << "Invalid dataset path: " << rootPath << "\n";
        std::cout.rdbuf(origCoutBuf);
        return -1;
    }

    std::cout << "Dataset path: " << rootPath << "\n";
    std::cout << "Mode: " << (showVisual ? "Text + Visual" : "Text Only") << "\n\n";

    std::vector<fs::path> sequenceDirs;
    for (const auto& entry : fs::recursive_directory_iterator(rootPath)) {
        if (entry.is_directory() && entry.path().filename() == "data") {
            sequenceDirs.push_back(entry.path().parent_path());
        }
    }
    std::sort(sequenceDirs.begin(), sequenceDirs.end());

    std::cout << "Extracting features from " << sequenceDirs.size() << " sequences...\n";
    std::vector<SequenceSample> samples;
    for (const auto& seq : sequenceDirs) {
        SequenceSample s;
        if (extractSequenceFeatures(seq, s)) {
            samples.push_back(s);
        }
    }

    int N = static_cast<int>(samples.size());
    if (N < 6) {
        std::cerr << "Insufficient sequences found for training/evaluation.\n";
        std::cout.rdbuf(origCoutBuf);
        return -1;
    }

    int numFeats = static_cast<int>(samples[0].features.size());
    int correct = 0;
    double totalIoU = 0.0;
    int confusionMatrix[7][7] = {0};

    // Leave-One-Out Cross Validation with In-Fold Feature Standard Scaling
    for (int i = 0; i < N; ++i) {
        // 1. Calculate training mean and standard deviation per feature
        std::vector<float> mean(numFeats, 0.0f);
        std::vector<float> stddev(numFeats, 0.0f);

        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            for (int f = 0; f < numFeats; ++f) {
                mean[f] += samples[j].features[f];
            }
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
            if (stddev[f] < 1e-6f) stddev[f] = 1.0f; // prevent zero division
        }

        // 2. Prepare standardized training matrices
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

        // 3. Train multi-class RBF Support Vector Machine
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setC(8.0);
        svm->setGamma(0.12);
        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 2000, 1e-6));

        svm->train(trainData, cv::ml::ROW_SAMPLE, trainLabels);

        // 4. Standardize and test on the held-out sample
        cv::Mat testSample(1, numFeats, CV_32F);
        for (int f = 0; f < numFeats; ++f) {
            testSample.at<float>(0, f) = (samples[i].features[f] - mean[f]) / stddev[f];
        }

        samples[i].predictedLabel = static_cast<int>(svm->predict(testSample));
        int trueLabel = samples[i].trueLabel;

        confusionMatrix[trueLabel][samples[i].predictedLabel]++;
        if (samples[i].predictedLabel == trueLabel) correct++;
        totalIoU += samples[i].iou;

        std::cout << "Seq: " << samples[i].seqName
                  << " | True: " << getActionName(trueLabel)
                  << " | Pred: " << getActionName(samples[i].predictedLabel)
                  << " | IoU (Frame 20): " << samples[i].iou << "\n";
    }

    std::cout << "\n================ EVALUATION METRICS ================\n";
    std::cout << "Total Processed: " << N << " sequences\n";
    std::cout << "Global Accuracy: " << (static_cast<double>(correct) / N) * 100.0 << "%\n";
    std::cout << "Mean IoU (mIoU): " << (totalIoU / N) << "\n\n";

    std::cout << "Confusion Matrix (Rows: Ground Truth, Cols: Predicted):\n";
    std::cout << "\tWALK\tJOG\tRUN\tBOX\tWAVE\tCLAP\n";
    for (int r = 1; r <= 6; ++r) {
        std::cout << getActionName(r).substr(0, 4) << "\t";
        for (int c = 1; c <= 6; ++c) {
            std::cout << confusionMatrix[r][c] << "\t";
        }
        std::cout << "\n";
    }

    if (showVisual) {
        std::cout << "\nStarting visual playback (Press 'q' or 'ESC' to skip)..." << std::endl;
        for (const auto& sample : samples) {
            renderSequenceVisual(sample);
        }
        cv::destroyAllWindows();
    }

    std::cout.rdbuf(origCoutBuf);
    return 0;
}