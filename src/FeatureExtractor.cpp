//Samuele Volpato
#include "FeatureExtractor.h"

float computeFrameCrossings(const std::vector<float>& widths) { //crossings*frame
    if (widths.size() < 4) 
        return 0.0f;

    float meanVal = std::accumulate(widths.begin(), widths.end(), 0) / widths.size();
    int crossings = 0;
    for (int i = 1; i < widths.size(); ++i) {
        if ((widths[i - 1] - meanVal) * (widths[i] - meanVal) < 0) {
            crossings++; // count every time we cross the average bbox width
        }
    }
    return static_cast<float>(crossings) / static_cast<float>(widths.size());
}

void extractFeatures(SequenceSample& sample, std::vector<cv::Point2f>& centroids, float H){
    double totalOverheadEnergy = 0;
    double totalChestMy = 0;
    double totalChestMx = 0;
    double totalConvergence = 0;
    double leftSideMx = 0;
    double rightSideMx = 0;
    double verticalFlowEnergy = 0;
    double horizontalFlowEnergy = 0;
    int chestCount = 0;
    int overheadCount = 0;
    double maxUpperWidth = 0;

    std::vector<float> wholeBodyKineticEnergies; // array of "quantity" of movement in bbox in each frame
    std::vector<float> widths; // all bboxes widths

    // fills arrays that we will need for future features
    widths.push_back(static_cast<float>(sample.bboxes[0].width));

    for (int i = 1; i < sample.frames.size(); ++i) {
        cv::Mat flow;
        // compute opticalflow, standard params except reduced winsize 15 -> 10 for detecting localized movements gives better results
        cv::calcOpticalFlowFarneback(sample.frames[i - 1], sample.frames[i], flow, 0.5, 3, 10, 3, 5, 1.2, 0);

        cv::Rect curBBox = sample.bboxes[i]; // update bbox

        // fills arrays that we will need for future features
        widths.push_back(static_cast<float>(curBBox.width));

        if (curBBox.width > maxUpperWidth) 
            maxUpperWidth = curBBox.width; // save max box width as feature

        int padX = static_cast<int>(curBBox.width * 0.10);
        int padY = static_cast<int>(curBBox.height * 0.10);

        // compute a new paddedBox with 10% padding
        cv::Rect paddedBBox(curBBox.x - padX, curBBox.y - padY, 
                            curBBox.width + 2 * padX, curBBox.height + 2 * padY);

        // prevent the top-left corner from going into negative coordinates
        paddedBBox.x = std::max(0, paddedBBox.x);
        paddedBBox.y = std::max(0, paddedBBox.y);

        // prevent the width and height from extending past the right and bottom edges
        paddedBBox.width = std::min(flow.cols - paddedBBox.x, paddedBBox.width);
        paddedBBox.height = std::min(flow.rows - paddedBBox.y, paddedBBox.height);
        
        cv::Rect fullBody = paddedBBox;

        float frameEnergy = 0.0f;
        if (fullBody.area() > 0) {
            for (int r = fullBody.y; r < fullBody.y + fullBody.height; ++r) { // examine every pixel of bbox
                for (int c = fullBody.x; c < fullBody.x + fullBody.width; ++c) {
                    cv::Point2f p = flow.at<cv::Point2f>(r, c);
                    float mag = std::hypot(p.x, p.y); // get magnitute of the movement of the pixel
                    if (mag > 0.5) { // avoid small movements (noise)
                        frameEnergy += mag; // add to the total energy of the frame
                    }
                }
            }
            wholeBodyKineticEnergies.push_back(frameEnergy);
        }

        // upper 20% of bbox taken as overhead
        cv::Rect overhead(paddedBBox.x, paddedBBox.y, paddedBBox.width, static_cast<int>(paddedBBox.height * 0.20));

        if (overhead.area() > 0) {
            for (int r = overhead.y; r < overhead.y + overhead.height; ++r) { // examine every pixel of overhead bbox
                for (int c = overhead.x; c < overhead.x + overhead.width; ++c) {
                    cv::Point2f p = flow.at<cv::Point2f>(r, c);
                    float mag = std::hypot(p.x, p.y); // get magnitute of the movement of the pixel
                    if (mag > 0.5) { // avoid small movements (noise)
                        totalOverheadEnergy += mag; // add to the total overhead energy
                        overheadCount++; // how many times we get movement overhead
                    }
                }
            }
        }

        // cut the top 28% of the bbox then take the 35% as the chest part
        cv::Rect chest(paddedBBox.x, paddedBBox.y + static_cast<int>(paddedBBox.height * 0.28),
                        paddedBBox.width, static_cast<int>(paddedBBox.height * 0.35));

        if (chest.area() > 0) {
            float midX = chest.x + chest.width / 2.0f;

            for (int r = chest.y; r < chest.y + chest.height; ++r) {
                for (int c = chest.x; c < chest.x + chest.width; ++c) {
                    cv::Point2f p = flow.at<cv::Point2f>(r, c);
                    float mag = std::hypot(p.x, p.y); // get magnitute of the movement of the pixel
                    if (mag > 0.5) {
                        totalChestMx += std::abs(p.x); // total movement in x
                        totalChestMy += std::abs(p.y); // total movement in y

                        float angle = std::abs(std::atan2(p.y, p.x)); // angle of movement
                        horizontalFlowEnergy += std::abs(std::cos(angle)) * mag; // scale angle x with magnitude
                        verticalFlowEnergy   += std::abs(std::sin(angle)) * mag; // scale angle y with magnitude

                        if (c < midX) {
                            totalConvergence += p.x; // sum for x convergence
                            leftSideMx += std::abs(p.x); // total movement at sx
                        } else {
                            totalConvergence -= p.x;
                            rightSideMx += std::abs(p.x); // total movement at dx
                        }
                        chestCount++; // how many times we get movement in chest
                    }
                }
            }
        }
    }

    // max x movement between frames
    float maxInstantMx = 0.0f;
    for (size_t i = 1; i < centroids.size(); ++i) {
        float instMx = std::abs(centroids[i].x - centroids[i - 1].x);
        if (instMx > maxInstantMx) 
            maxInstantMx = instMx;
    }
    float normMaxInstantMx = maxInstantMx / H; // normalize with H

    // x displacement
    float netTranslationX = (centroids.size() > 1) ?
        (std::abs(centroids.back().x - centroids.front().x) / H) : 0.0f;

    float kineticP90 = 0.0f; // take kinetic energy at 90% to avoid outliers
    float kineticBurstRatio = 1.0f; // ratio as P50 / P95
    if (!wholeBodyKineticEnergies.empty()) {
        std::sort(wholeBodyKineticEnergies.begin(), wholeBodyKineticEnergies.end());
        int idx50 = static_cast<int>(wholeBodyKineticEnergies.size() * 0.50);
        int idx90 = static_cast<int>(wholeBodyKineticEnergies.size() * 0.90);
        int idx95 = static_cast<int>(wholeBodyKineticEnergies.size() * 0.95);

        kineticP90 = wholeBodyKineticEnergies[idx90] / H;
        float medianEn = std::max(0.1f, wholeBodyKineticEnergies[idx50]);
        kineticBurstRatio = wholeBodyKineticEnergies[idx95] / medianEn;
    }

    float frameCrossings = computeFrameCrossings(widths); // cross mean width per frame

    // mean of centroid Y
    float meanCentroidY = 0.0f;
    for (auto& c : centroids) 
        meanCentroidY += c.y;
    meanCentroidY /= centroids.size();
    
    // variance of centroid Y
    float varCentroidY = 0.0f;
    for (auto& c : centroids) {
        varCentroidY += (c.y - meanCentroidY) * (c.y - meanCentroidY);
    }
    varCentroidY /= centroids.size();
    float normStdCentroidY = std::sqrt(varCentroidY) / H; // standard deviation scaled by H

    // scale by H and by count the features computated in the frames loop
    float normOverheadEnergy = (overheadCount > 0) ? static_cast<float>((totalOverheadEnergy / overheadCount) / H) : 0.0f;
    float normChestMy = (chestCount > 0) ? static_cast<float>((totalChestMy / chestCount) / H) : 0.0f;
    float normChestMx = (chestCount > 0) ? static_cast<float>((totalChestMx / chestCount) / H) : 0.0f;
    float normConvergence = (chestCount > 0) ? static_cast<float>((totalConvergence / chestCount) / H) : 0.0f;
    float normMaxUpperWidth = static_cast<float>(maxUpperWidth / H);

    float angularRatio = (horizontalFlowEnergy + 1e-4f) / (verticalFlowEnergy + 1e-4f); // avoid 0 by summing small numb
    // movement right-left symmetry
    float symmetryRatio = (leftSideMx + rightSideMx > 0.1) ?
        static_cast<float>((2.0 * std::min(leftSideMx, rightSideMx)) / (leftSideMx + rightSideMx)) : 0.0f;

    // save features
    sample.features = {
        netTranslationX,
        normMaxInstantMx,
        kineticP90,
        kineticBurstRatio,
        frameCrossings,
        normOverheadEnergy,
        normChestMy,
        normChestMx,
        normConvergence,
        normMaxUpperWidth,
        angularRatio,
        symmetryRatio,
        normStdCentroidY,
    };
}