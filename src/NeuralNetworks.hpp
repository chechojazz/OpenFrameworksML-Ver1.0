//
//  NeuralNetworks.hpp
//  OscMl-ver3_0
//
//  Created by Ignasi Nou Plana on 6/3/18.
//

#ifndef NeuralNetworks_hpp
#define NeuralNetworks_hpp

#include <stdio.h>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

void trainWithANN(Ptr<ANN_MLP>& ann,Ptr<TrainData>& trainData);
float predictWithANN(Ptr<ANN_MLP>& lr,vector<float> featuresToPredictWith);

#endif /* NeuralNetworks_hpp */
