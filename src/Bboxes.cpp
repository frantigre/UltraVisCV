#include "Bboxes.h"


void findBoxes(std::vector<cv::Mat> video, SequenceSample& sample, std::vector<cv::Rect>& bboxes, float& H){

    //NOTE VIDEO IS ALREADY IN GRAYSCALE!!!
    
    int imgW = video[0].cols;
    int imgH = video[0].rows;

    cv::Mat energy = computeTemporalEnergy(video);
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

    std::vector<cv::Point2f> centroids;
    std::vector<float> actorHeights;
    
    //std::cout<<"PER FRAME"<<std::endl;

    for (size_t i=0; i<video.size(); i++) {
        int prevI = std::max(0,static_cast<int>(i)-1);
        int succI = std::min(video.size()-1, i+1);
        cv::Mat diff1=difference(video[i],video[prevI]);
        cv::threshold(diff1, diff1, 10, 255, cv::THRESH_BINARY); //could be changed 10 since it's already in difference
        cv::Mat diff2=difference(video[i],video[succI]);
        cv::threshold(diff2, diff2, 10, 255, cv::THRESH_BINARY); // "
        cv::Mat motionMask;
        cv::bitwise_and(diff1,diff2,motionMask);
        /*
        cv::Mat diff1,diff2,motionMask;
        cv::absdiff(video[i], video[prevI], diff1);
        cv::absdiff(video[i], video[succI], diff2);
        cv::bitwise_and(diff1, diff2, motionMask);                          //forse da girare se il senso è tenere solo dove c'è stato il movimento sia prima che dopo
        cv::threshold(motionMask, motionMask, 10, 255, cv::THRESH_BINARY);*/
        
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
                }*/
            }
        }
        
        if(bestScoreB*0.6<=secBest){
            curBox=curBox|secBox;
        }
        
        //laplacian part
        cv::Mat smooth,edges;
        cv::GaussianBlur(video[i], smooth, cv::Size(5,5), 0);
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
        cv::cvtColor(video[i], debug, cv::COLOR_GRAY2BGR);
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
        }*/
        
        if (curBox.area()>0) {
            
            /*cv::Rect safeBox = curBox & cv::Rect(0, 0, imgW, imgH); 
            float density = 0.0f;
            if (safeBox.area() > 0) {
                int movingPixels = cv::countNonZero(motionMask(safeBox));
                density = static_cast<float>(movingPixels) / safeBox.area();
            }*/
            
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
                curBox.width = std::min(imgW - curBox.x, minW);*/
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

    sample.bbox20=bboxes[19];
    std::sort(actorHeights.begin(), actorHeights.end());
    H = std::max(30.0f, actorHeights[actorHeights.size() / 2]); // pick median H for scale inv.
    return;    
}
