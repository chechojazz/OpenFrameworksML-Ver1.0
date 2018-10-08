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

void saveDataRecordedToLocalMemory(vector<ofFile> vectorOfFiles,vector<vector<Mat>> modelsData, vector<vector<Mat>> modelsLabel){    //save recorded data to local memory directory
    
    for(int numModels = 0; numModels < vectorOfFiles.size(); ++numModels){
        
        using boost::filesystem::path;
        
        String nameModel = "dataModelNumber";
        nameModel = nameModel + std::to_string(numModels);
        nameModel = nameModel + ".csv";
        path pathname(nameModel.c_str());
        
        vectorOfFiles[numModels].open(pathname,ofFile::WriteOnly);
        
        //START DATA ADQUISITION AND DATA DUMP
        for (int numSavedData = 0; numSavedData < modelsData[numModels].size(); ++numSavedData ){
            
            for (int n = 0; n <= modelsData[numModels][numSavedData].rows-1;++n){
                
                for (int m = 0; m <= modelsData[numModels][numSavedData].cols-1;++m){
                    
                    vectorOfFiles[numModels] << modelsData[numModels][numSavedData].at<float>(n,m) << " , " ;
                    
                }
                vectorOfFiles[numModels] << modelsLabel[numModels][numSavedData].at<float>(n,0); // (n,0) -> 0 Perque sol te una columna
                vectorOfFiles[numModels] << "\n";
            }
        }
        cout << "[Recorded Data] - Model number "<< numModels+1 << " save as a CSV file" << endl;
        //END OF DATA DUMP
        vectorOfFiles[numModels].close();
        //---------------- 11/07/2018 Ignasi: Va, los datos volcados son aquellos que se han grabado, por lo tanto hay que tener en cuenta que si se van grabando modelos y no se elimina lo grabado i luego se hace un volcado se haran de todos los datos y no solo del último! Y SOLO SE HACE EL VOLCADO DE ESOS MODELOS QUE ESTAN ACTIVADOS (SE PUEDE CAMBIAR!!)
        
    }
}

void saveTrainedData(vector<Ptr<ANN_MLP>> ann){
    
    using boost::filesystem::path;
    
    char * dir = getcwd(NULL, 0); // Platform-dependent, see reference link below
    printf("[Ann Files Parameters] - Files are going to be saved in:  %s \n", dir);
    
    for(int i = 0; i < ann.size(); i++){
        
        String nameModel = "annParameters";
        nameModel = nameModel + std::to_string(i+1);
        nameModel = nameModel + ".xml";
        path pathname(nameModel.c_str());
        
        FileStorage fs;
        fs.open(nameModel, FileStorage::WRITE);
        
        if (!fs.isOpened()){
            cout << "[Ann Files Parameters] - File Not opened" << endl;
        }else{
            cout << "[Ann Files Parameters] - File opended" <<endl;
            if(ann[i].empty() == false){
                ann[i]->write(fs);
                cout << "[Ann Files Parameters] - Ann Variable Model number " << i+1 <<" saved!.\n" << endl;
            }else{
                cout << "[Ann Files Parameters] - Ann Variable Model number " << i+1 <<" is empty.\n" << endl;
            }
            fs.release();
        }
    }
}
void loadTrainingData(vector<Ptr<ANN_MLP>>& ann){
    
    //Open the Open File Dialog
    ofFileDialogResult openFileResult= ofSystemLoadDialog("Select a XML File");
    
    //Check if the user opened a file
    if (openFileResult.bSuccess){
        
        //We have a file, check it and process it
        processOpenFileSelection(openFileResult,ann);
        
    }else {
        cout <<"[Load Training from Doc] - User hit cancel" << endl;;
    }
}

void processOpenFileSelection(ofFileDialogResult openFileResult,vector<Ptr<ANN_MLP>>& ann){
    
    //    ofLogVerbose("getName(): "  + openFileResult.getName());
    //    ofLogVerbose("getPath(): "  + openFileResult.getPath());
    
    ofFile file (openFileResult.getPath());
    
    if (file.exists()){
        
        //        ofLogVerbose("The file exists - now checking the type via file extension");
        string fileExtension = ofToUpper(file.getExtension());
        
        if(fileExtension == "XML"){
            
            FileStorage opencv_file(openFileResult.getPath(), FileStorage::READ);
            
            Ptr<ANN_MLP> nn = ANN_MLP::create();
            nn->read(opencv_file.root());
            
            cout << openFileResult.getName()<< endl;
            
            char lastCharName = openFileResult.getName().at(openFileResult.getName().length() - 5);
            
            if (isdigit(lastCharName) == true){
                
                int numberModelInt = lastCharName-48;
                if((numberModelInt < 6) && (numberModelInt > 0)){
                    
                    ann[numberModelInt - 1] = nn;
                    cout << "[Load Training from Doc] - Model number "<< numberModelInt <<" updated." << endl;
                    
                }else{
                    cout << "[Load Training from Doc] - Models from 1 to 5" << endl;
                }
            }else{
                cout << "[Load Training from Doc] - Please specify the model number in the name of the file 'modelAnnNumber_.xml'; e.g. 'modelAnnNumber1.xml'" << endl;
            }
            opencv_file.release();
        }else{
            cout << "[Load Training from Doc] - The extension must be .xml and the structure in accordance with the OpenCV storage document." << endl;
        }
    }
}


