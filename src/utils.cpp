//
//  utils.cpp
//  OscMl-ver3_0
//
//  Created by Ignasi Nou Plana on 15/2/18.
//

#include "utils.hpp"
#include "NeuralNetworks.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

void saveDataRecorded(Mat &data, vector<Mat> &label, vector<float> &features, vector<float> &isSoundGood, vector<bool> &boolVecModelsActive){
    Mat vectorRow = Mat(features);
//    cout << vectorRow << endl;
    data.push_back(vectorRow.t());
    
    for ( int i = 0; i <= isSoundGood.size()-1; ++i){
        
        if (boolVecModelsActive[i]==1){
            label[i].push_back(isSoundGood[i]);
            label[i].convertTo(label[i], CV_32F);
        }
    }
    data.convertTo(data,CV_32F);
}

void createTrainData(vector<vector<Mat>> dataVecMat, vector<vector<Mat>> labelVecMat, vector<bool> modelsAct,vector<bool> methodAct ,vector<Ptr<TrainData>>& trainDataOfModels){
    
    cout << "Recorded data is transforming to training data..." <<endl;
    //    cout << methodAct[0] << endl;
    //    cout << methodAct[1] << endl;
    
    Mat auxForTrainData,auxForTrainLabel;
    Mat trainClasses;
    
    for (int n = 0; n<=4; ++n){ //PUSH BACK DE TOTES LES DADES DELS DIFERENTS MODELS ACTIVATS DE TOTES LES GRABACIONS!
        if(modelsAct[n]==1){
            if( (dataVecMat[n].empty()==true) || (labelVecMat[n].empty()==true) ){
                cout << "Data/label matrix " << n+1 << " is empty, cannot train." << endl;
            }else{
                for(int m = 0; m<=dataVecMat[n].size()-1; ++m){
                    auxForTrainData.push_back(dataVecMat[n][m]);
                    auxForTrainLabel.push_back(labelVecMat[n][m]);
                }
                
                //_______________Tractament de els labels segons métode utiltizat!__________________________________________________________
                
                if (methodAct[1] == true){ //If SVM is active we need a CV_32S mat type!!!
                    auxForTrainLabel.convertTo(auxForTrainLabel, CV_32SC1);
                }
                
                if(methodAct[2] == 1){
                    cout << "Fixing ANN data"  << endl; // if ANN one hot encoded label structure!
                    vector<float> array;
                    
                    if (auxForTrainLabel.isContinuous()) { //GUARRRADA transformar mat amb vector per aprofitar el codi de trainClasses!
                        array.assign((float*)auxForTrainLabel.datastart, (float*)auxForTrainLabel.dataend);
                    } else {
                        for (int i = 0; i < auxForTrainLabel.rows; ++i) {
                            array.insert(array.end(), auxForTrainLabel.ptr<float>(i), auxForTrainLabel.ptr<float>(i)+auxForTrainLabel.cols);
                        }
                    }//GUARRRADA!
                    trainClasses = Mat::zeros( auxForTrainLabel.rows, 2, CV_32FC1 ); // Primera columna coses mal! segona columna coses bien!!!
                    
                    for( int i = 0; i < trainClasses.rows; i++ )
                    {
                        trainClasses.at<float>(i, array[i]) = 1.f;
                    }
                    auxForTrainLabel = trainClasses;
//                    cout << auxForTrainLabel << endl;

                }
                //__________________________________________________________________________________________________________________________________
                
                
                trainDataOfModels[n] = TrainData::create(auxForTrainData, ROW_SAMPLE, auxForTrainLabel); // pensar de distingir les dades per metodes i per models!
                trainClasses.release();
                auxForTrainData.release();
                auxForTrainLabel.release();
            }
        }
    }
}

