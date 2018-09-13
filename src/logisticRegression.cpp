//
//  logisticRegression.cpp
//  OscMl-ver3_0
//
//  Created by Ignasi Nou Plana on 7/2/18.
//

#include "logisticRegression.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

static float sigmoid(float x)
{
    return   (x / (1 + abs(x)));
}

Ptr<LogisticRegression> createLogisticRegression(){
    
    Ptr <LogisticRegression> lr = LogisticRegression::create();
    return lr;
}

Mat trainWithLR(Ptr<LogisticRegression>& lr,Ptr<TrainData>& trainData,float learningRate, int itereations){
    
    //    cout << trainData->getTrainSamples() << endl;
    //    cout << trainData->getTrainResponses() << endl;
    //    lr->setLearningRate(learningRate);
    //    lr->setIterations(itereations);
    //    lr->setRegularization(LogisticRegression::REG_L2);
    //    lr->setTrainMethod(LogisticRegression::BATCH);
    //    lr->setMiniBatchSize(1);
    lr = createLogisticRegression();
    try{
        cout << "training" << endl;
        lr->train(trainData);
        
    }
    catch (cv::Exception& e){
        const char* err_msg = e.what();
        //        cout << err_msg << endl; //Per mirar lerror!!
        
        cout << "2 dif class or more examples" << endl;
    }
    Mat thetas = lr->get_learnt_thetas();
    //    cout << trainData->getResponses() << endl;
    trainData.release();
    return thetas;
    
}

float predictWithLR(Ptr<LogisticRegression>& lr,vector<float> featuresToPredictWith){ //TODO
    
    Mat result;
    Mat featuresMat;
    Mat thetasLR;
    float probFloat;
    featuresMat.push_back(featuresToPredictWith); //????
    featuresMat.convertTo(featuresMat, CV_32F);
    //
    //    cout << featuresMat.t() << endl;
    if (lr.empty()){
        cout << "ERROR empty lr" << endl;
    }else{
        thetasLR = lr->get_learnt_thetas();
        try{
            lr->predict(featuresMat.t(),result);
        }catch (cv::Exception& e){
            const char* err_msg = e.what();
            cout << err_msg << endl; //Per mirar lerror!!
            
        }
        featuresToPredictWith.insert(featuresToPredictWith.begin(),1); //Add cont variable to predict for the % ONLY!
        Mat prob = thetasLR*Mat(featuresToPredictWith);
        probFloat = prob.at<float>(0,0);
        //    cout << sigmoid(probFloat) << endl; //PROB! btw -1 and 1
        
        return sigmoid(probFloat);
    }
    return probFloat;
    
}
//----------------TODO----------------------------
void saveClassifierLR(Ptr<LogisticRegression>& lr){
    
    const String saveFileNameLR = "LR_TrainedData";
    cout << "saving the classifier to " << saveFileNameLR << endl;
    lr->save(saveFileNameLR);
}
//----------------TODO----------------------------




