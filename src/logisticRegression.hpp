//
//  logisticRegression.hpp
//  OscMl-ver3_0
//
//  Created by Ignasi Nou Plana on 7/2/18.
//

#ifndef logisticRegression_hpp
#define logisticRegression_hpp

#include <stdio.h>
#include <opencv2/ml.hpp>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::ml;
//public lr variable to acces to predictions
//Ptr <LogisticRegression> lr = LogisticRegression::create();

Ptr<LogisticRegression> createLogisticRegression();

Mat trainWithLR(Ptr<LogisticRegression>& lr,Ptr<TrainData>& trainData,float learningRate, int itereations);
float predictWithLR(Ptr<LogisticRegression>& lr,vector<float> featuresToPredictWith);
void saveClassifierLR(Ptr<LogisticRegression>& lr);

#endif /* logisticRegression_hpp */

