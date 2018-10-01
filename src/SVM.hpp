//
//  SVM.hpp
//  OscMl-ver3_0
//
//  Created by Ignasi Nou Plana on 15/2/18.
//

#ifndef SVM_hpp
#define SVM_hpp

#include <stdio.h>
#include <opencv2/ml.hpp>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::ml;

Mat trainWithSVM(Ptr<SVM>& svm,Ptr<TrainData>& trainData);


#endif /* SVM_hpp */
