//Francesco Ariani
#include "Performance.h"

// center the text given the width
std::string centerText(std::string text, int width) {
    if (text.length() >= static_cast<size_t>(width)) {
        return text;
    }
    int padding = width - text.length();
    int padLeft = padding / 2;
    int padRight = padding - padLeft;
    return std::string(padLeft, ' ') + text + std::string(padRight, ' ');
}

cv::Mat ConfMatrix(std::vector<SequenceSample> samples){
    cv::Mat matrix = cv::Mat::zeros(6,6,CV_32S); //6 classes by definition
    for(size_t i=0; i<samples.size(); i++){
        matrix.at<int>(samples[i].trueLabel-1, samples[i].predictedLabel-1)++;  //they need to be 0-5 so could be needed a -1
    }
    return matrix;
}

float F1Score(cv::Mat matrix, int picked_class){
    int index=picked_class-1;
    //std::cout<<matrix.at<int>(index,index)<<std::endl;
    int TP=matrix.at<int>(index,index);
    if(TP==0) return 0.0f; //no true positive, no need to compute anything + avoid division by 0
    int FP=0,FN=0; //TN not needed for F1
    for(int i=0; i<6;i++){ //6 classes by definition
        if(i!=index){
        FP+=matrix.at<int>(i,index);
        FN+=matrix.at<int>(index,i);
        }
    }
    float precision=static_cast<float>(TP)/(TP+FP);
    float recall=static_cast<float>(TP)/(TP+FN);
    return 2*precision*recall/(precision+recall);
}

void EvaluateModel(std::vector<SequenceSample> samples){
    std::string output="";
    cv::Mat confusion = ConfMatrix(samples);
    output+="Confusion Matrix (Rows: Ground Truth, Cols: Predicted):\n";
    std::vector<std::string> types = {"boxing", "handclapping", "handwaving", "running", "jogging", "walking" };
    
    // Using the custom helper function instead of std::format
    output += centerText("", 14);
    
    for(size_t i=0; i<types.size(); i++){
        output += centerText(types[i], 14);
    }
    output+="\n";
    
    for(int i=0; i<confusion.rows; i++){
        output += centerText(types[i], 14);
        for(int j=0; j<confusion.cols; j++){
            // Convert integer to string before passing to centerText
            output += centerText(std::to_string(confusion.at<int>(i,j)), 14);
        }
        output+="\n";
    }
    output+="\n";
    
    for(size_t i=0; i<types.size(); i++){        
        float F1tmp=F1Score(confusion, i+1);
        output+="F1 score for class "+types[i]+": "+std::to_string(F1tmp)+"\n";
    }
    output+="\n";
    float mIoU=0;
    int accuracy=0;
    for(size_t i=0; i<samples.size(); i++){
        mIoU+=samples[i].iou;
        saveAnnotatedFrame20(samples[i],"output"); //writes file visualizing bbox and category of frame 20
        if((samples[i].trueLabel==samples[i].predictedLabel)&&(samples[i].trueLabel!=-1)){   //second check is in case of some error in reading labels, should never be -1 trueLabel
            accuracy+=1;
        }
    }
    
    output+="global accuracy: "+std::to_string(static_cast<float>(accuracy)/samples.size())+"\n";
    output+="mIoU: "+std::to_string(mIoU/samples.size())+"\n";
    std::cout<<output<<std::endl;       //flush is technically more correct
    std::ofstream file("output/Metrics.txt");     //the directory is created by saveAnnotatedFrame20 if it doesn't exist
    if(file.is_open()){
        file<<output;
        file.close();
    }
}
