#include "Performance.h"

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
    //std::cout<<confusion<<std::endl;
    //to fix
    //output+=confusion;
    for(int i=1; i<=6; i++){        //maybe an enum?
        float F1tmp=F1Score(confusion, i);
        //std::cout<<"F1 score for class "<<i<<": "<<F1tmp<<std::endl;
        output+="F1 score for class "+std::to_string(i)+": "+std::to_string(F1tmp)+"\n";
    }
    float mIoU=0;
    int accuracy=0;
    for(size_t i=0; i<samples.size(); i++){
        mIoU+=samples[i].iou;
        saveAnnotatedFrame20(samples[i],"prova"); //writes file visualizing bbox and category of frame 20
        if((samples[i].trueLabel==samples[i].predictedLabel)&&(samples[i].trueLabel!=-1)){   //second check is in case of some error in reading labels, should never be -1 trueLabel
            accuracy+=1;
        }
    }
    
    //std::cout<<"accuracy: "<<static_cast<float>(accuracy)/samples.size()<<std::endl;
    //std::cout<<"mIoU: "<<mIoU/samples.size()<<std::endl;
    output+="accuracy: "+std::to_string(static_cast<float>(accuracy)/samples.size())+"\n";
    output+="mIoU: "+std::to_string(mIoU/samples.size())+"\n";
    std::cout<<output<<std::endl;       //flush is technically more correct
}



/*void EvaluateModel() {
    //TODO: adattare con funzioni di Fra
    for (int r = 1; r <= 6; ++r) {
        std::cout << getActionName(r).substr(0, 4) << "\t";

        for (int c = 1; c <= 6; ++c) {
            std::cout << ConfMatrix[r][c] << "\t";
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
        int tp = ConfMatrix[c][c];
        int fn = 0;
        int fp = 0;

        for (int j = 1; j <= 6; ++j) {
            if (j != c) {
                fn += ConfMatrix[c][j];
                fp += ConfMatrix[j][c];
            }
        }

        double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1 = (precision + recall > 1e-6) ? 2.0 * (precision * recall) / (precision + recall) : 0.0;

        macroF1 += f1;

        std::cout << std::left << std::setw(14) << getActionName(c)
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << precision
                  << std::setw(12) << recall
                  << std::setw(12) << f1 << "\n";
    }

    std::cout << "--------------------------------------------------\n";
    std::cout << "Macro Average F1-Score: " << std::fixed << std::setprecision(4) << (macroF1 / 6.0) << "\n";

    std::cout << "\nSaved " << samples.size() << " annotated frame-20 images to: "
              << fs::absolute(outImgDir).string() << "\n";
}*/
