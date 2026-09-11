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
        case HANDWAVING: return "waving";
        case HANDCLAPPING: return "clapping";
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
    std::string frame20Path;
    cv::Rect bbox20;
    std::vector<float> features;
    float netTranslationX = 0.0f;
    int trueLabel = -1;
    int predictedLabel = -1;
    double iou = 0.0;
};

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

float computeStrideCadence(const std::vector<float>& signal) {
    if (signal.size() < 4) return 0.0f;
    float meanVal = std::accumulate(signal.begin(), signal.end(), 0.0f) / signal.size();
    int crossings = 0;
    for (size_t i = 1; i < signal.size(); ++i) {
        if ((signal[i - 1] - meanVal) * (signal[i] - meanVal) < 0.0f) {
            crossings++;
        }
    }
    return static_cast<float>(crossings) / static_cast<float>(signal.size());
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

    std::vector<std::string> framePaths;
    if (fs::exists(dataDir) && fs::is_directory(dataDir)) {
        for (const auto& entry : fs::directory_iterator(dataDir)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".png" || ext == ".bmp" || ext == ".jpeg") {
                framePaths.push_back(entry.path().string());
            }
        }
    }
    std::sort(framePaths.begin(), framePaths.end());
    if (framePaths.size() < 20) return false;

    sample.frame20Path = framePaths[19];

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

    int nFrames = static_cast<int>(framePaths.size());
    std::vector<cv::Mat> grays(nFrames);
    for (int i = 0; i < nFrames; ++i) {
        cv::Mat bgr = cv::imread(framePaths[i]);
        cv::cvtColor(bgr, grays[i], cv::COLOR_BGR2GRAY);
    }

    int imgW = grays[0].cols;
    int imgH = grays[0].rows;

    cv::Mat energy = computeTemporalEnergy(grays);
    cv::Mat energyMask;
    cv::threshold(energy, energyMask, 8, 255, cv::THRESH_BINARY);
    cv::morphologyEx(energyMask, energyMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 21)));

    std::vector<std::vector<cv::Point>> eContours;
    cv::findContours(energyMask, eContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect globalActorZone(0, 0, imgW, imgH);
    double maxEArea = 0;
    for (const auto& c : eContours) {
        cv::Rect r = cv::boundingRect(c);
        float ar = static_cast<float>(r.width) / r.height;
        if (r.area() > maxEArea && r.area() > 250 && ar < 2.2f) {
            maxEArea = r.area();
            globalActorZone = r;
        }
    }

    std::vector<cv::Rect> bboxes;
    std::vector<cv::Point2f> centroids;
    std::vector<float> aspectRatios;
    std::vector<float> widths;
    std::vector<float> actorHeights;

    for (int i = 0; i < nFrames; ++i) {
        int prevIdx = std::max(0, i - 1);
        int nextIdx = std::min(nFrames - 1, i + 1);

        cv::Mat diff1, diff2, motionMask;
        cv::absdiff(grays[i], grays[prevIdx], diff1);
        cv::absdiff(grays[i], grays[nextIdx], diff2);
        cv::bitwise_and(diff1, diff2, motionMask);
        cv::threshold(motionMask, motionMask, 10, 255, cv::THRESH_BINARY);
        cv::morphologyEx(motionMask, motionMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 15)));

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(motionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Rect curBox(0, 0, 0, 0);
        double bestScore = 1e9;
        for (const auto& c : contours) {
            double a = cv::contourArea(c);
            if (a > 60.0) {
                cv::Rect r = cv::boundingRect(c);
                float ar = static_cast<float>(r.width) / r.height;
                double score = std::abs(ar - 0.42f) * 100.0 - std::min(a, 3000.0) * 0.05;
                if (score < bestScore) {
                    bestScore = score;
                    curBox = r;
                }
            }
        }

        if (curBox.area() > 0) {
            if (curBox.height < globalActorZone.height * 0.72f) {
                int newH = globalActorZone.height;
                int newY = std::max(0, std::min(curBox.y, globalActorZone.y));
                if (newY + newH > imgH) newH = imgH - newY;
                curBox.y = newY;
                curBox.height = newH;
            }
            int minW = static_cast<int>(curBox.height * 0.30f);
            if (curBox.width < minW) {
                int cx = curBox.x + curBox.width / 2;
                curBox.x = std::max(0, cx - minW / 2);
                curBox.width = std::min(imgW - curBox.x, minW);
            }
            bboxes.push_back(curBox);
            centroids.push_back(cv::Point2f(curBox.x + curBox.width / 2.0f, curBox.y + curBox.height / 2.0f));
            aspectRatios.push_back(static_cast<float>(curBox.width) / curBox.height);
            widths.push_back(static_cast<float>(curBox.width));
            actorHeights.push_back(static_cast<float>(curBox.height));
        } else if (!bboxes.empty()) {
            bboxes.push_back(bboxes.back());
            centroids.push_back(centroids.back());
            aspectRatios.push_back(aspectRatios.back());
            widths.push_back(widths.back());
            actorHeights.push_back(actorHeights.back());
        } else {
            bboxes.push_back(globalActorZone);
            centroids.push_back(cv::Point2f(globalActorZone.x + globalActorZone.width / 2.0f, globalActorZone.y + globalActorZone.height / 2.0f));
            aspectRatios.push_back(static_cast<float>(globalActorZone.width) / globalActorZone.height);
            widths.push_back(static_cast<float>(globalActorZone.width));
            actorHeights.push_back(static_cast<float>(globalActorZone.height));
        }
    }

    // 5-Frame Temporal Rolling Centroid Anchor around Frame 20 for stable mIoU
    cv::Rect smoothedBox20 = bboxes[19];
    if (bboxes.size() >= 23) {
        int avgX = 0, avgY = 0, avgW = 0, avgH = 0;
        for (int k = 17; k <= 21; ++k) {
            avgX += bboxes[k].x;
            avgY += bboxes[k].y;
            avgW += bboxes[k].width;
            avgH += bboxes[k].height;
        }
        smoothedBox20 = cv::Rect(avgX / 5, avgY / 5, avgW / 5, avgH / 5);
    }
    sample.bbox20 = smoothedBox20;

    std::sort(actorHeights.begin(), actorHeights.end());
    float H = std::max(30.0f, actorHeights[actorHeights.size() / 2]);

    double totalOverheadEnergy = 0.0;
    double totalTorsoVy = 0.0;
    double totalTorsoVx = 0.0;
    double totalConvergence = 0.0;
    double leftSideVx = 0.0;
    double rightSideVx = 0.0;
    double verticalFlowEnergy = 0.0;
    double horizontalFlowEnergy = 0.0;
    int flowCount = 0;
    int overheadCount = 0;
    double maxUpperWidth = 0.0;

    std::vector<float> wholeBodyKineticEnergies;

    for (int i = 1; i < nFrames; ++i) {
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(grays[i - 1], grays[i], flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        cv::Rect curBBox = bboxes[i];
        if (curBBox.width > maxUpperWidth) maxUpperWidth = curBBox.width;

        cv::Rect fullROI = curBBox & cv::Rect(0, 0, flow.cols, flow.rows);
        if (fullROI.area() > 0) {
            for (int r = fullROI.y; r < fullROI.y + fullROI.height; ++r) {
                for (int c = fullROI.x; c < fullROI.x + fullROI.width; ++c) {
                    cv::Point2f f = flow.at<cv::Point2f>(r, c);
                    float m = std::hypot(f.x, f.y);
                    if (m > 0.5f) wholeBodyKineticEnergies.push_back(m);
                }
            }
        }

        // Upper 32% Bounding Area
        cv::Rect overheadROI(curBBox.x, curBBox.y, curBBox.width, static_cast<int>(curBBox.height * 0.32));
        overheadROI &= cv::Rect(0, 0, flow.cols, flow.rows);

        // Mid-Torso 35% Bounding Area
        cv::Rect chestROI(curBBox.x, curBBox.y + static_cast<int>(curBBox.height * 0.28),
                          curBBox.width, static_cast<int>(curBBox.height * 0.35));
        chestROI &= cv::Rect(0, 0, flow.cols, flow.rows);

        if (overheadROI.area() > 0) {
            for (int r = overheadROI.y; r < overheadROI.y + overheadROI.height; ++r) {
                for (int c = overheadROI.x; c < overheadROI.x + overheadROI.width; ++c) {
                    cv::Point2f f = flow.at<cv::Point2f>(r, c);
                    float mag = std::hypot(f.x, f.y);
                    if (mag > 0.4f) {
                        totalOverheadEnergy += mag;
                        overheadCount++;
                    }
                }
            }
        }

        if (chestROI.area() > 0) {
            float midX = chestROI.x + chestROI.width / 2.0f;
            for (int r = chestROI.y; r < chestROI.y + chestROI.height; ++r) {
                for (int c = chestROI.x; c < chestROI.x + chestROI.width; ++c) {
                    cv::Point2f f = flow.at<cv::Point2f>(r, c);
                    float mag = std::hypot(f.x, f.y);
                    if (mag > 0.5f) {
                        totalTorsoVx += std::abs(f.x);
                        totalTorsoVy += std::abs(f.y);

                        float angle = std::abs(std::atan2(f.y, f.x));
                        horizontalFlowEnergy += std::abs(std::cos(angle)) * mag;
                        verticalFlowEnergy   += std::abs(std::sin(angle)) * mag;

                        if (c < midX) {
                            totalConvergence += f.x;
                            leftSideVx += std::abs(f.x);
                        } else {
                            totalConvergence -= f.x;
                            rightSideVx += std::abs(f.x);
                        }
                        flowCount++;
                    }
                }
            }
        }
    }

    // Displacement & Net Vector Translation
    float totalDisplacementX = 0.0f;
    float maxInstantVx = 0.0f;
    for (size_t i = 1; i < centroids.size(); ++i) {
        float instVx = std::abs(centroids[i].x - centroids[i - 1].x);
        totalDisplacementX += instVx;
        if (instVx > maxInstantVx) maxInstantVx = instVx;
    }
    float normVelocityX = (totalDisplacementX / centroids.size()) / H;
    float normMaxInstantVx = maxInstantVx / H;

    // Endpoint Drift across the sequence
    sample.netTranslationX = (centroids.size() > 1) ?
        (std::abs(centroids.back().x - centroids.front().x) / H) : 0.0f;

    // Kinetic Energy Spectrum: 90th percentile and Burst Ratio (P95 / P50)
    float kineticP90 = 0.0f;
    float kineticBurstRatio = 1.0f;
    if (!wholeBodyKineticEnergies.empty()) {
        std::sort(wholeBodyKineticEnergies.begin(), wholeBodyKineticEnergies.end());
        size_t idx50 = static_cast<size_t>(wholeBodyKineticEnergies.size() * 0.50);
        size_t idx90 = static_cast<size_t>(wholeBodyKineticEnergies.size() * 0.90);
        size_t idx95 = static_cast<size_t>(wholeBodyKineticEnergies.size() * 0.95);

        kineticP90 = wholeBodyKineticEnergies[idx90] / H;
        float medianFlow = std::max(0.1f, wholeBodyKineticEnergies[idx50]);
        kineticBurstRatio = wholeBodyKineticEnergies[idx95] / medianFlow;
    }

    float strideCadence = computeStrideCadence(widths);
    float meanAR = std::accumulate(aspectRatios.begin(), aspectRatios.end(), 0.0f) / aspectRatios.size();
    float varAR = 0.0f;
    for (float ar : aspectRatios) varAR += (ar - meanAR) * (ar - meanAR);
    varAR /= aspectRatios.size();

    float normOverheadEnergy = (overheadCount > 0) ? static_cast<float>((totalOverheadEnergy / overheadCount) / H) : 0.0f;
    float normTorsoVy = (flowCount > 0) ? static_cast<float>((totalTorsoVy / flowCount) / H) : 0.0f;
    float normTorsoVx = (flowCount > 0) ? static_cast<float>((totalTorsoVx / flowCount) / H) : 0.0f;
    float normConvergence = (flowCount > 0) ? static_cast<float>((totalConvergence / flowCount) / H) : 0.0f;
    float normMaxUpperWidth = static_cast<float>(maxUpperWidth / H);

    float angularRatio = (horizontalFlowEnergy + 1e-4f) / (verticalFlowEnergy + 1e-4f);
    float symmetryRatio = (leftSideVx + rightSideVx > 0.1) ?
        static_cast<float>((2.0 * std::min(leftSideVx, rightSideVx)) / (leftSideVx + rightSideVx)) : 0.0f;

    // 13 Multimodal Spatial-Temporal Descriptors
    sample.features = {
        normVelocityX,          // 0: Mean Frame Translation / H
        sample.netTranslationX, // 1: Net Endpoint Traversal / H
        normMaxInstantVx,       // 2: Peak Frame Jump / H
        kineticP90,             // 3: 90th Percentile Energy / H
        kineticBurstRatio,      // 4: Kinetic Burst (Running vs Jogging)
        strideCadence,          // 5: Aspect Ratio Frequency
        varAR,                  // 6: Gait Deformation Variance
        normOverheadEnergy,     // 7: Flow Above Shoulders / H
        normTorsoVy,            // 8: Vertical Flow / H
        normTorsoVx,            // 9: Horizontal Flow / H
        normConvergence,        // 10: Inward Midline Convergence / H
        normMaxUpperWidth,      // 11: Arm Reach / H
        angularRatio,           // 12: Directional Angle (Clapping vs Waving)
        symmetryRatio           // 13: Bilateral Symmetry
    };

    if (hasGT && nFrames >= 20) {
        int W = imgW;
        int imgH_ref = imgH;
        double gx_c = gt.bbox.x;
        double gy_c = gt.bbox.y;
        double gw = gt.bbox.width;
        double gh = gt.bbox.height;

        if (gx_c <= 1.0 && gy_c <= 1.0 && gw <= 1.0 && gh <= 1.0) {
            gx_c *= W; gy_c *= imgH_ref; gw *= W; gh *= imgH_ref;
        }
        int gx = static_cast<int>(gx_c - gw / 2.0);
        int gy = static_cast<int>(gy_c - gh / 2.0);
        sample.iou = computeIoU(sample.bbox20, cv::Rect(gx, gy, static_cast<int>(gw), static_cast<int>(gh)));
    } else {
        sample.iou = 0.0;
    }

    return true;
}

void saveAnnotatedFrame20(const SequenceSample& sample, const fs::path& outDir) {
    cv::Mat frame = cv::imread(sample.frame20Path);
    if (frame.empty()) return;

    cv::Rect bbox = sample.bbox20;
    std::string labelText = getActionName(sample.predictedLabel);

    cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);

    cv::Point textPos;
    if (bbox.x > 80) {
        textPos = cv::Point(std::max(10, bbox.x - 75), std::max(20, bbox.y + 15));
    } else {
        textPos = cv::Point(std::min(frame.cols - 80, bbox.x + bbox.width + 5), std::max(20, bbox.y + 15));
    }

    cv::putText(frame, labelText, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 255), 2);

    fs::path outFile = outDir / (sample.seqName + "_frame20.png");
    cv::imwrite(outFile.string(), frame);
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

    fs::path rootPath = (argc >= 2) ? argv[1] : "../dataset";
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
        if (netTraverse > 0.35f && isStationaryPrediction) {
            pred = WALKING;
        }
        // Guardrail 2: Stationary actors cannot be classified as locomotion
        if (netTraverse < 0.12f && isLocomotionPrediction) {
            // Re-route back to dominant stationary prediction
            pred = (samples[i].features[7] > 0.02f) ? HANDWAVING : HANDCLAPPING;
        }

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
}