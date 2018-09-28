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
#include "ofApp.h"


using namespace std;
using namespace cv;
using namespace cv::ml;

//Recording process
void saveDataRecorded(Mat &data, vector<Mat> &label, vector<float> &features, vector<float> &isSoundGood, vector<bool> &boolVecModelsActive);
void createTrainData(vector<vector<Mat>> dataVecMat, vector<vector<Mat>> labelVecMat, vector<bool> modelsAct,vector<bool> methodAct,vector<Ptr<TrainData>>& trainDataOfModels);


//Storage process
void saveDataRecordedToLocalMemory(vector<ofFile> vectorOfFiles, vector<vector<Mat>> modelsData, vector<vector<Mat>> modelsLabel, vector<string> paramlist, string pathAndFileName); //save recorded data to local memory directory
void saveTrainedData(vector<Ptr<ANN_MLP>> ann, vector<string> paramlist, string pathAndFileName);
void loadTrainingData(vector<Ptr<ANN_MLP>>& ann, string pathAndFileName, string strModelNum);

#endif /* utils_hpp */
