#include "Bboxes.h"


void findBoxes(std::vector<cv::Mat> video, SequenceSample& sample, std::vector<cv::Rect>& bboxes, float& H){

    //NOTE VIDEO IS ALREADY IN GRAYSCALE!!!
    
    int imgW = video[0].cols;
    int imgH = video[0].rows;

    cv::Mat mDiff = MeanOfDifferences(video);
    cv::Mat mDiffMask;
    cv::threshold(mDiff, mDiffMask, 4, 255, cv::THRESH_BINARY); //remove too small values, probably noise
    cv::morphologyEx(mDiffMask, mDiffMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 21))); //finds area where the actor has passed

    std::vector<std::vector<cv::Point>> dContours;
    cv::findContours(mDiffMask, dContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //finds contours where the actor has passed
    
    cv::Rect personZoneVideo(0, 0, imgW, imgH);
    double maxEArea = 0;
    
    
    for (size_t i=0; i<dContours.size(); i++) {
        cv::Rect tmp = cv::boundingRect(dContours[i]);
        //if (tmp.area() > maxEArea && tmp.area() > 100) {  //take the biggest area     SE PEGGIORA RIMETTERE QUESTO
        if (tmp.area() > maxEArea) {  //take the biggest area
            maxEArea = tmp.area();                                   
            personZoneVideo = tmp;
        }
    }

    //movement part
    std::vector<cv::Point2f> centroids;
    std::vector<float> actorHeights;
    
    //std::cout<<"PER FRAME"<<std::endl;
    
    for (size_t i=0; i<video.size(); i++) {
        int prevI = std::max(0,static_cast<int>(i)-1);
        int succI = std::min(video.size()-1, i+1);      //compute difference between current, previous and next frame; excpet first and last
        cv::Mat diff1=difference(video[i],video[prevI]);
        cv::threshold(diff1, diff1, 10, 255, cv::THRESH_BINARY); 
        cv::Mat diff2=difference(video[i],video[succI]);
        cv::threshold(diff2, diff2, 10, 255, cv::THRESH_BINARY); 
        cv::Mat motionMask;
        cv::bitwise_and(diff1,diff2,motionMask);       //logical and since diffs are thresholded = finds area where there was movement in both frames
        
        cv::morphologyEx(motionMask, motionMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 15)));

        std::vector<std::vector<cv::Point>> mContours;
        cv::findContours(motionMask, mContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);        //finds contours where there was movement in both frames
        
        
        
        cv::Rect curBox(0,0,0,0), secBox(0,0,0,0);   //current BEST Box = best box found so far for this frame
        double bestScoreB = 0,secBest=0; //for "new part"
        for (size_t j=0; j<mContours.size(); j++) {
            double area = cv::contourArea(mContours[j]);
            double proximityX = 0, proximityY=0; //proximity with previous box
            double similarityW = 1, similarityH = 1; //similarity with previous box, to check due to expansions
            if (area > 10 && area/static_cast<double>(imgW*imgH)<0.8) { //avoid too small areas and too big ones
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
                //if(i==19) std::cout<<"proximity "<<proxXFactor*proxYFactor<<"  similarity "<<simFactor<<"  ARFactor "<<ARFactor<<std::endl;
                double score = (0.4*area/static_cast<double>(imgW*imgH)+0.6*proxXFactor*proxYFactor)*simFactor*ARFactor;        //computes score based on area, distance from bbox in previous frame,       
                                                                                                                                //similarity in width and height in prev. frame and aspect ratio
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
            }
        }
        
        if(bestScoreB*0.6<=secBest){        //if two boxes are close in score computes union of the two (doesn't work very well though)
            curBox=curBox|secBox;
        }
        
        //laplacian part
        cv::Mat smooth,edges;
        cv::GaussianBlur(video[i], smooth, cv::Size(5,5), 0);
        cv::Laplacian(smooth, edges, CV_16S, 3);
        cv::convertScaleAbs(edges,edges);
        cv::threshold(edges,edges,30,255,cv::THRESH_BINARY);        //obtain remove noise in laplacian
        cv::morphologyEx(edges,edges,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5)));
        //removes horizontal noise = baseboard inside
        cv::Mat hNoise=edges.clone();           
        cv::morphologyEx(hNoise,hNoise,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,5)));
        cv::morphologyEx(hNoise,hNoise,cv::MORPH_OPEN,cv::getStructuringElement(cv::MORPH_RECT, cv::Size(static_cast<int>(imgW*0.4), 1))); //finds very long horizontal elements
        edges.setTo(0,hNoise);                                                                                                             //removes horizontal elements found
        std::vector<std::vector<cv::Point>> lContours;
        cv::findContours(edges, lContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);     //computes contours from element given by laplacian
        
        cv::Rect curEdge(0,0,0,0);
        double bestScoreE=0;
        //std::cout<<lContours.size()<<" position "<<i<<std::endl;
        for (size_t j=0; j<lContours.size(); j++) {
            double area = cv::contourArea(lContours[j]);
            if(area>15){
                cv::Rect proposal = cv::boundingRect(lContours[j]);
                double proximity=std::sqrt(std::pow((proposal.x+proposal.width/2.0f)-(curBox.x+curBox.width/2.0f),2)+               //computes distance between box found previously and box given by
                                           std::pow((proposal.y+proposal.height/2.0f)-(curBox.y+curBox.height/2.0f),2));            //laplacian element
                double score=0.85*area/static_cast<double>(imgW*imgH)+0.15*Gaussian(proximity,0,0,20,60);   //assign score to laplacian element based on area and distance from box found by movements
                if(score > bestScoreE){                    
                    bestScoreE=score;
                    curEdge=proposal;
                }                     
            }
        }
        curBox=curBox|curEdge;      //union between box found by movement part and laplacian part
        
        if (curBox.area()>0) {
            if (curBox.height<personZoneVideo.height*0.7) { //expands vertically if box is too small
                int newH=personZoneVideo.height;
                int newY=std::max(0, std::min(curBox.y, personZoneVideo.y));
                if (newY+newH>imgH) newH=imgH-newY;
                curBox.y=newY;
                curBox.height=newH;
            }
            int minW = static_cast<int>(curBox.height*0.3); //expands horizontally if box is too small
            if (curBox.width < minW) {
                double newCent= 0.7*(curBox.x+curBox.width/2) + 0.3*(personZoneVideo.x+personZoneVideo.width/2);
                curBox.width=minW;
                curBox.x=std::max(0,static_cast<int>(newCent-minW/2));
                if (curBox.x+curBox.width>imgW) curBox.x=imgW-curBox.width;
            }
            bboxes.push_back(curBox);
            centroids.push_back(cv::Point2f(curBox.x+curBox.width/2.0f, curBox.y+curBox.height/2.0f));   //saves found bbox
            actorHeights.push_back(static_cast<float>(curBox.height));
        } else if (bboxes.size()>0) {
            bboxes.push_back(bboxes[i-1]);        //no bbox found, takes the previous
            centroids.push_back(centroids[i-1]);
            actorHeights.push_back(actorHeights[i-1]);
        } else {
            bboxes.push_back(personZoneVideo);           //no bbox found AND no previous box => saves the whole mean = movement
            centroids.push_back(cv::Point2f(personZoneVideo.x + personZoneVideo.width / 2.0f, personZoneVideo.y + personZoneVideo.height / 2.0f));
            actorHeights.push_back(static_cast<double>(personZoneVideo.height));
        }
    }

    sample.bbox20=bboxes[19];
    std::sort(actorHeights.begin(), actorHeights.end());        //needed for feature extraction
    H = std::max(30.0f, actorHeights[actorHeights.size() / 2]); // pick median H for scale inv.
    return;    
}
