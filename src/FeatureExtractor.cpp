#include "FeatureExtractor.h"

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

void extractFeatures(SequenceSample& sample, std::vector<cv::Mat>& grays, std::vector<cv::Rect>& bboxes, float H){
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
    std::vector<float> widths;
    std::vector<float> aspectRatios;
    std::vector<cv::Point2f> centroids;

    widths.push_back(static_cast<float>(bboxes[0].width));
    aspectRatios.push_back(static_cast<float>(bboxes[0].width) / bboxes[0].height);
    centroids.push_back(cv::Point2f(bboxes[0].x + bboxes[0].width / 2.0f, bboxes[0].y + bboxes[0].height / 2.0f));

    for (int i = 1; i < grays.size(); ++i) {
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(grays[i - 1], grays[i], flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        cv::Rect curBBox = bboxes[i];
        if (curBBox.width > maxUpperWidth) maxUpperWidth = curBBox.width;

        // 1. Define a 10% padding margin based on the current box size
        int padX = static_cast<int>(curBBox.width * 0.10);
        int padY = static_cast<int>(curBBox.height * 0.10);

        // 2. Create the expanded box and safely keep it inside the image boundaries
        cv::Rect paddedBBox(curBBox.x - padX, curBBox.y - padY, 
                            curBBox.width + 2 * padX, curBBox.height + 2 * padY);
        paddedBBox &= cv::Rect(0, 0, flow.cols, flow.rows);
        
        widths.push_back(static_cast<float>(curBBox.width));
        aspectRatios.push_back(static_cast<float>(curBBox.width) / curBBox.height);
        centroids.push_back(cv::Point2f(curBBox.x + curBBox.width / 2.0f, curBBox.y + curBBox.height / 2.0f));

        // 3. Use paddedBBox for all your flow extractions instead of curBBox
        cv::Rect fullROI = paddedBBox; 
        
        float currentFrameEnergy = 0.0f;
        if (fullROI.area() > 0) {
            for (int r = fullROI.y; r < fullROI.y + fullROI.height; ++r) {
                for (int c = fullROI.x; c < fullROI.x + fullROI.width; ++c) {
                    cv::Point2f f = flow.at<cv::Point2f>(r, c);
                    float mag = std::hypot(f.x, f.y);
                    // Threshold at 0.5f to ignore minor background noise/compression artifacts
                    if (mag > 0.5f) {
                        currentFrameEnergy += mag;
                    }
                }
            }
            wholeBodyKineticEnergies.push_back(currentFrameEnergy);
        }

        // Upper 32% Bounding Area (using padded dimensions)
        cv::Rect overheadROI(paddedBBox.x, paddedBBox.y, paddedBBox.width, static_cast<int>(paddedBBox.height * 0.32));
        overheadROI &= cv::Rect(0, 0, flow.cols, flow.rows);

        // Mid-Torso 35% Bounding Area (using padded dimensions)
        cv::Rect chestROI(paddedBBox.x, paddedBBox.y + static_cast<int>(paddedBBox.height * 0.28),
                        paddedBBox.width, static_cast<int>(paddedBBox.height * 0.35));
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

    // NEW: Vertical Centroid Oscillation (Bounce)
    float meanCentroidY = 0.0f;
    for (const auto& c : centroids) meanCentroidY += c.y;
    meanCentroidY /= centroids.size();
    
    float varCentroidY = 0.0f;
    for (const auto& c : centroids) {
        varCentroidY += (c.y - meanCentroidY) * (c.y - meanCentroidY);
    }
    varCentroidY /= centroids.size();
    float normStdCentroidY = std::sqrt(varCentroidY) / H; // Standard deviation scaled by height

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
        symmetryRatio,          // 13: Bilateral Symmetry
        //normStdCentroidY,       // 14: Vertical Bounce (Running vs Jogging)
    };
}