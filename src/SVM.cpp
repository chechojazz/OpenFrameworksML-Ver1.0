//
//  SVM.cpp
//  OscMl-ver3_0
//
//  Created by Ignasi Nou Plana on 15/2/18.
//

#include "SVM.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;


Mat trainWithSVM(Ptr<SVM>& svm,Ptr<TrainData>& trainData){
    try{
        cout << trainData->getTrainSamples() << endl;
        cout << trainData->getTrainResponses() << endl;
        
        svm->train(trainData->getTrainSamples(),ROW_SAMPLE,trainData->getTrainResponses());
        cout << "training SVM OK" << endl;
    }
    catch (cv::Exception& e){
        const char* err_msg = e.what();
        cout << err_msg << endl;
        cout << "This means that your 2 'different classes' have the same label, your classification can not be done!" << endl;
    }
    Mat svmSupportVectors;
    Mat alpha;
//    svmSupportVectors = svm->getSupportVectors();
    svm->getDecisionFunction(0,alpha,svmSupportVectors);
    if( svmSupportVectors.empty()==true){
//        cout << "SupportVectors Mat is Empty! This probably means that your 2 'different classes' have the same label, your classification can not be done!" << endl;
    }
    cout << "Alpha values of SVM: " << alpha << endl;
    return svmSupportVectors;
}
