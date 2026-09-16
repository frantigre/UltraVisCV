#ifndef MODEL_EVALUATOR_H
#define MODEL_EVALUATOR_H

#include <vector>
#include "SequenceSample.h"

class ModelEvaluator {
public:
    void DeploySVM(std::vector<SequenceSample>& samples);
    void EvaluateModel();

private:
    int numFeats;
    int correct;
    double totalIoU;
    int confusionMatrix[7][7];
};

#endif