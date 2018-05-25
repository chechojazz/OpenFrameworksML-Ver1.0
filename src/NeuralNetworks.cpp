//
//  NeuralNetworks.cpp
//  OscMl-ver3_0
//
//  Created by Ignasi Nou Plana on 6/3/18.
//

#include "NeuralNetworks.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

void trainWithANN(Ptr<ANN_MLP>& ann,Ptr<TrainData>& trainData){
    
    ann = ANN_MLP::create();
    int numFeatures = trainData->getSamples().cols;
    
    Mat layer_size( 1, 3, CV_32SC1 );
    
    layer_size.at<int>(0) = numFeatures; //input layers NUMERO DE FEATURES = NUMERO DE INPUT LAYERS
    layer_size.at<int>(1) = floor(numFeatures/2); //Hiiden layer num of nodes
    layer_size.at<int>(2) = 2; //Output layer numero de classes a classificar
    
    ann->setLayerSizes(layer_size);
    ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, FLT_EPSILON));
    ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001);
    ann->train(trainData);
//    cout << ann->getWeights(4) << endl;
}

float predictWithANN(Ptr<ANN_MLP>& ann,vector<float> featuresToPredictWith){
    
    int size = featuresToPredictWith.size();
    Mat featuresMat(size,1, CV_32F),result;
    
    float probFloat;
    
    //featuresMat.push_back(featuresToPredictWith); //????
    //cout<<"t:"<<featuresMat.type()<<"\nd:"<<featuresMat.depth()<<"\n";//to debug
    //featuresMat.convertTo(featuresMat, CV_32F);
    
    
    for(int i = 0; i < size; i++) {
        featuresMat.at<float>(i,0) = featuresToPredictWith[i];
    }
    

//    cout << featuresMat.t() << endl;
    
    if (ann.empty()){
        cout << "ERROR empty ANN" << endl;
        return probFloat = 0;
    }else{
        try{
            ann->predict(featuresMat.t(),result);
        }catch (cv::Exception& e){
            const char* err_msg = e.what();
            cout << err_msg << endl; //Per mirar lerror!!
        }
        cout << result << endl;
        
        probFloat = result.at<float>(0, 1);
        return probFloat;
    }
    
}
