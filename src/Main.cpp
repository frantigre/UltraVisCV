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


namespace fs = std::filesystem;

/*bool extractSequenceFeatures(const fs::path& seqDir, SequenceSample& sample) {      //da rimuovere
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
    
    //std::cout<<"NEW PART"<<std::endl;
    
    int imgW = grays[0].cols;
    int imgH = grays[0].rows;

    cv::Mat energy = computeTemporalEnergy(grays);
    cv::Mat energyMask;
    //cv::threshold(energy, energyMask, 8, 255, cv::THRESH_BINARY);
    cv::threshold(energy, energyMask, 4, 255, cv::THRESH_BINARY);
    cv::morphologyEx(energyMask, energyMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 21))); //trova l'area in cui la persona è passata

    std::vector<std::vector<cv::Point>> eContours;
    cv::findContours(energyMask, eContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //trova i contours = i bordi tra la zona in cui è passata la persona e dove non è passata

    cv::Rect personZoneVideo(0, 0, imgW, imgH);
    double maxEArea = 0;
    
    
    for (size_t i=0; i<eContours.size(); i++) {
        cv::Rect tmp = cv::boundingRect(eContours[i]);
        //float ar = static_cast<float>(r.width) / r.height;
        if (tmp.area() > maxEArea && tmp.area() > 100) {  //prende il contour più grosso, probabilmente va tolto ar perchè se la persona corre l'area è più larga che alta
            maxEArea = tmp.area();                                   //forse anche abbassare 250
            personZoneVideo = tmp;
        }
    }

    std::vector<cv::Rect> bboxes;
    std::vector<cv::Point2f> centroids;
    std::vector<float> actorHeights;
    
    //std::cout<<"PER FRAME"<<std::endl;

    for (size_t i=0; i<grays.size(); i++) {
        int prevI = std::max(0,static_cast<int>(i)-1);
        int succI = std::min(grays.size()-1, i+1);
        cv::Mat diff1=difference(grays[i],grays[prevI]);
        cv::threshold(diff1, diff1, 10, 255, cv::THRESH_BINARY); //could be changed 10 since it's already in difference
        cv::Mat diff2=difference(grays[i],grays[succI]);
        cv::threshold(diff2, diff2, 10, 255, cv::THRESH_BINARY); // "
        cv::Mat motionMask;
        cv::bitwise_and(diff1,diff2,motionMask);
        /*
        cv::Mat diff1,diff2,motionMask;
        cv::absdiff(grays[i], grays[prevI], diff1);
        cv::absdiff(grays[i], grays[succI], diff2);
        cv::bitwise_and(diff1, diff2, motionMask);                          //forse da girare se il senso è tenere solo dove c'è stato il movimento sia prima che dopo
        cv::threshold(motionMask, motionMask, 10, 255, cv::THRESH_BINARY);
        
        cv::morphologyEx(motionMask, motionMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 15)));

        std::vector<std::vector<cv::Point>> mContours;
        cv::findContours(motionMask, mContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        
        
        cv::Rect curBox(0,0,0,0), secBox(0,0,0,0);   //current BEST Box = best box found so far for this frame
        //cv::Point2f curCent,secCent;
        double bestScoreB = 0,secBest=0; //for "new part"
        //double bestScoreB = 1e9;
        for (size_t j=0; j<mContours.size(); j++) {
            double area = cv::contourArea(mContours[j]);
            //curCent.x=mContours[j].x + mContours[j].width / 2.0f;
            //curCent.y=mContours[j].y + mContours[j].height / 2.0f;
            double proximityX = 0, proximityY=0; //proximity with previous box
            double similarityW = 1, similarityH = 1; //similarity with previous box, to check due to expansions
            if (area > 10 && area/static_cast<double>(imgW*imgH)<0.8) { //avoid too small areas and too big ones
                //make the score something like "area + proximity to previous IF present * scale factor (too vertical or too horizontal close to 0 = no human form)
                
               
                cv::Rect proposal = cv::boundingRect(mContours[j]);
                if(i!=0){
                    if(bboxes[i-1].area()>0){
                        proximityX = centroids[i-1].x-(proposal.x+proposal.width/2.0f);
                        proximityY = centroids[i-1].y-(proposal.y+proposal.height/2.0f);
                        similarityW = static_cast<double>(proposal.width)/bboxes[i-1].width;
                        similarityH = static_cast<double>(proposal.height)/bboxes[i-1].height;
                    }
                }
                double proxXFactor = Gaussian(proximityX, -30, -10, 10, 30); //guassian with plateau withing -10, 10 (pixels of movement)
                double proxYFactor = Gaussian(proximityY, -20, -5, 5, 20); //guassian with plateau withing -5, 5 (")
                double simFactor = Gaussian(similarityW, 0.16, 0.5, 2, 6)*Gaussian(similarityH, 0.25, 0.66, 1.66, 4);    //box in new frame should be around same size as prevoius
                double ARFactor = Gaussian(static_cast<double>(proposal.width)/proposal.height,0.07,0.17,0.75,1.5);                  //median aspect ratio of human should be around 1:3, 1:4 when still
                //double score = (area+proximityX*proxXFactor+proximityY*proxYFactor)*simFactor*ARFactor;
                //if(i==19) std::cout<<"proximity "<<proxXFactor*proxYFactor<<"  similarity "<<simFactor<<"  ARFactor "<<ARFactor<<std::endl;
                double score = (0.4*area/static_cast<double>(imgW*imgH)+0.6*proxXFactor*proxYFactor)*simFactor*ARFactor;
                if(score > bestScoreB){
                    secBest=bestScoreB;
                    secBox=curBox;
                    
                    bestScoreB=score;
                    curBox=proposal;
                }
                else if(score > secBest){
                    secBest=score;
                    secBox=proposal;
                }
            /*
                
                cv::Rect tmp = cv::boundingRect(mContours[j]);
                float ar = static_cast<float>(tmp.width) / tmp.height;
                double score = std::abs(ar - 0.42f) * 100.0 - std::min(area, 3000.0) * 0.05; //da rivedere questo score sia per le proporzioni che per l'area
                if (score < bestScoreB) {
                    bestScoreB = score;
                    curBox = tmp;
                }
            }
        }
        
        if(bestScoreB*0.6<=secBest){
            curBox=curBox|secBox;
        }
        
        //laplacian part
        cv::Mat smooth,edges;
        cv::GaussianBlur(grays[i], smooth, cv::Size(5,5), 0);
        cv::Laplacian(smooth, edges, CV_16S, 3);
        cv::convertScaleAbs(edges,edges);
        cv::threshold(edges,edges,30,255,cv::THRESH_BINARY);
        cv::morphologyEx(edges,edges,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5)));
        cv::Mat hNoise=edges.clone();
        cv::morphologyEx(hNoise,hNoise,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,5)));
        cv::morphologyEx(hNoise,hNoise,cv::MORPH_OPEN,cv::getStructuringElement(cv::MORPH_RECT, cv::Size(static_cast<int>(imgW*0.4), 1)));
        edges.setTo(0,hNoise);
        std::vector<std::vector<cv::Point>> lContours;
        cv::findContours(edges, lContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        cv::Rect curEdge(0,0,0,0);
        double bestScoreE=0;
        //std::cout<<lContours.size()<<" position "<<i<<std::endl;
        for (size_t j=0; j<lContours.size(); j++) {
            double area = cv::contourArea(lContours[j]);
            if(area>15){
                cv::Rect proposal = cv::boundingRect(lContours[j]);
                double proximity=std::sqrt(std::pow((proposal.x+proposal.width/2.0f)-(curBox.x+curBox.width/2.0f),2)+
                                           std::pow((proposal.y+proposal.height/2.0f)-(curBox.y+curBox.height/2.0f),2));
                double score=0.85*area/static_cast<double>(imgW*imgH)+0.15*Gaussian(proximity,0,0,20,60);
                //std::cout<<"loop infinito"<<std::endl;
                if(score > bestScoreE){                    
                    bestScoreE=score;
                    curEdge=proposal;
                }                     
            }
        }
        curBox=curBox|curEdge;
        
        /*cv::Mat debug;
        cv::cvtColor(grays[i], debug, cv::COLOR_GRAY2BGR);
        if(i==19){
            /*std::cout<<"\ncontours found "<<mContours.size()<<"\n";
            cv::rectangle(debug, secBox, cv::Scalar(255,0,0),2);
            cv::rectangle(debug, curBox, cv::Scalar(0,0,255),2);
            cv::rectangle(debug, curBox|secBox, cv::Scalar(0,255,0),2);
            std::cout<<"best "<<bestScore<<" second "<<secBest<<"\n\n"<<std::endl;
            cv::imshow("squares", debug);*/
            //cv::imshow("laplac",edges);
            //cv::imshow("horiz",hNoise);
            /*cv::rectangle(debug, curBox, cv::Scalar(0,0,255),2);
            cv::rectangle(debug, curEdge, cv::Scalar(255,0,0),2);
            cv::imshow("two boxes", debug);
            //cv::waitKey(0);
        }
        
        if (curBox.area()>0) {
            
            /*cv::Rect safeBox = curBox & cv::Rect(0, 0, imgW, imgH); 
            float density = 0.0f;
            if (safeBox.area() > 0) {
                int movingPixels = cv::countNonZero(motionMask(safeBox));
                density = static_cast<float>(movingPixels) / safeBox.area();
            }
            
            if (curBox.height<personZoneVideo.height*0.7) { //expands vertically
                int newH=personZoneVideo.height;
                int newY=std::max(0, std::min(curBox.y, personZoneVideo.y));
                if (newY+newH>imgH) newH=imgH-newY;
                curBox.y=newY;
                curBox.height=newH;
            }
            int minW = static_cast<int>(curBox.height*0.3); //expands horizontally
            if (curBox.width < minW) {
                double newCent= 0.7*(curBox.x+curBox.width/2) + 0.3*(personZoneVideo.x+personZoneVideo.width/2);
                curBox.width=minW;
                curBox.x=std::max(0,static_cast<int>(newCent-minW/2));
                if (curBox.x+curBox.width>imgW) curBox.x=imgW-curBox.width;
                
                
                /*int cx = curBox.x + curBox.width / 2;
                curBox.x = std::max(0, cx - minW / 2);      //BEFORE
                curBox.width = std::min(imgW - curBox.x, minW);
            }
            bboxes.push_back(curBox);
            centroids.push_back(cv::Point2f(curBox.x+curBox.width/2.0f, curBox.y+curBox.height/2.0f));   //saves found bbox
            actorHeights.push_back(static_cast<float>(curBox.height));
        } else if (bboxes.size()>0) {
            bboxes.push_back(bboxes[i-1]);        //no bbox found, takes the previous
            centroids.push_back(centroids[i-1]);
            actorHeights.push_back(actorHeights[i-1]);
        } else {
            bboxes.push_back(personZoneVideo);           //no bbox found AND no previous box => saves the whole energy = movement
            centroids.push_back(cv::Point2f(personZoneVideo.x + personZoneVideo.width / 2.0f, personZoneVideo.y + personZoneVideo.height / 2.0f));
            actorHeights.push_back(static_cast<double>(personZoneVideo.height));
        }
    }
    //std::cout<<"END MY PART for now"<<std::endl;
    sample.bbox20=bboxes[19];

    std::sort(actorHeights.begin(), actorHeights.end());

    float H = std::max(30.0f, actorHeights[actorHeights.size() / 2]); // pick median H for scale inv.
    

        //SAMU FUNCTION
            
    extractFeatures(sample, grays, bboxes, H);
        
        
        //it's just IoU
        
    if (hasGT && grays.size() >= 20) {
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
}*/

/*void saveAnnotatedFrame20(const SequenceSample& sample, const fs::path& outDir) {
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
}*/

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
    std::vector<SequenceSample> samples;
    for (const auto& seq : sequenceDirs) {
        SequenceSample s;
        if (extractSequenceFeatures(seq, s)) {
            samples.push_back(s);
        }
    }

    int numFeats = static_cast<int>(samples[0].features.size());
    int correct = 0;
    double totalIoU = 0.0;
    int confusionMatrix[7][7] = {0};
    
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
}*/
int main(int argc, char** argv) {
    std::vector<std::vector<cv::Mat>> database;
    std::vector<std::vector<cv::Mat>> DBGray; //might not be needed
    std::vector<GroundTruth> gt;
    getDataset(database, gt);
    //std::cout<<"CARICA (credo)"<<std::endl;
    float mIoU=0;
    for(size_t i=0; i<database.size(); i++){
        std::cout<<"computing video "<<i+1<<std::endl;
        //DBGray.push_back(toGray(database[i])); not necessary probably
        std::vector<cv::Mat> videoG=toGray(database[i]);
        SequenceSample sample;
        std::vector<cv::Rect> bboxes;
        float H;
        
        sample.frame20=videoG[19]; //19 index = frame 20
        sample.seqName="video"+std::to_string(i);
        
        findBoxes(videoG, sample, bboxes, H);
        extractFeatures(sample, videoG, bboxes, H);
        
        sample.iou = computeIoU(sample.bbox20, gt[i].bbox);
        /*std::cout<<"\nvideo "<<i+1<<"\nbox20 "<<sample.bbox20<<std::endl;
        std::cout<<"grounTruth "<<gt[i].bbox<<std::endl;
        std::cout<<"IoU video "<<i+1<<" : "<<sample.iou<<std::endl;*/
        mIoU+=sample.iou;
        saveAnnotatedFrame20(sample,"prova"); 
    }
    std::cout<<mIoU/database.size()<<std::endl;
}
