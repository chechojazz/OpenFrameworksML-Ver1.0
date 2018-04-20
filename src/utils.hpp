//
//  utils.hpp
//  OscMl-ver3_0
//
//  Created by Ignasi Nou Plana on 15/2/18.
//

#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <opencv2/ml.hpp>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::ml;

void saveDataRecorded(Mat &data, vector<Mat> &label, vector<float> &features, vector<float> &isSoundGood, vector<bool> &boolVecModelsActive);
void createTrainData(vector<vector<Mat>> dataVecMat, vector<vector<Mat>> labelVecMat, vector<bool> modelsAct,vector<bool> methodAct,vector<Ptr<TrainData>>& trainDataOfModels);
#endif /* utils_hpp */
