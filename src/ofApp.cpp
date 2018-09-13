#include "ofApp.h"


// ---------- NUM OF FEATURES --------------------------
int numFeatures = 58;

// ----------------- DATA INPUT ---------------------
Mat dataMatrix, recorded;
vector<Mat> labelMatrix(5);
vector<vector<Mat>> modelsData(5), modelsLabel(5); //Number of models
vector<float> isSoundGood(5);
vector<float> featuresViolinRT(numFeatures); // wek/inputs
vector<Mat> thetas(5);

// ----------------------  Control Inputs! ----------------------------

float startRecording = 0;
float startTraining = 0;
float startRunning = 0;

// ----------- METHODS ACTIVATED ------------------

bool boolLogisticRegression=false;
bool boolSVM = false;
bool boolANN = true;

// ------------------- Machine Learning ----------------------------

vector<Ptr<TrainData>> trainDataOfModels(5); // one train data for each model (Could be diferent)


// ------ Logistic Regression -----------
vector<Ptr<LogisticRegression>> lr(5);

// ------- SVM -----------------
Ptr<SVM> svm = SVM::create(); //Deprecated
vector<Mat> svmVectors(5); //Deprecated

// -------- ANN ------------
vector<Ptr<ANN_MLP>> ann(5);

//-----------ModelsActivatedRecording(Save data!!!)----------------------

vector<int> modelsActivated(5); // Only for data converting purpose
vector<int> modelsDesactivated(5); // Only for data converting purpose
vector<int> auxModelAct(5);
int numModelsActivated = 5; //Models 1, 2, 3, 4, 5 - DEFAULT VIOLINRT
vector<bool> boolVecModelsActive={1,1,1,1,1}; // Which models are activated to record

//-----------ModelsActivatedRunning(To predict!)----------------------

vector<int> modelsActivatedRunning(5);
vector<int> modelsDesactivatedRunning(5);
int numModelsActivatedRunning = 5;
vector<bool> boolVecModelsActiveRunning={1,1,1,1,1}; // Which models are activated to record

// ---------- Methods Activated -----------------------------
vector<bool> boolVecMethodsActive={boolLogisticRegression,boolSVM,boolANN}; // Which methods are activated

//  ----------------------------- Send Output Vector of floats  -----------------------------
vector<float> senderOutput = {0,0,0,0,0};

//--------------------------------------------------------------
void ofApp::setup(){
    receiver.setup(6448);
    sender.setup("localhost", 12000);
}

//--------------------------------------------------------------
void ofApp::update(){
    ofSetLogLevel(OF_LOG_VERBOSE);
    
    while(receiver.hasWaitingMessages()){
        
        ofxOscMessage msg;
        ofxOscMessage sendMsg;
        
        sendMsg.setAddress("/wek/outputs");
        receiver.getNextMessage(msg);
        
        if(msg.getAddress()=="/wek/inputs"){
            //            cout << "data" << endl;
            for (int i = 0; i <= numFeatures-1; ++i){ /// SI TENEMOS POCOS DATOS NO ENTRENA!!!!
                if (isnan(msg.getArgAsFloat(i)) == false){
                    featuresViolinRT[i] = msg.getArgAsFloat(i);
                }
                //cout << "FEATURE NUMBER: " << i << " " <<featuresViolinRT[i] << endl;
            }
        }
        if(msg.getAddress()=="/wekinator/control/outputs"){ // Labels de cada frame para cada modelo, isSoundGood tiene size 5 por los 5 modelos
            for (int j = 0; j <= isSoundGood.size()-1; ++j){
                isSoundGood[j] = msg.getArgAsFloat(j);
                //                              cout << isSoundGood[j] << endl;
            }
        }
        if (msg.getAddress()=="/wekinator/control/startRecording"){
            cout << "start recoording" << endl;
            startRecording = 1;
        }
        if (msg.getAddress()=="/wekinator/control/stopRecording"){
            cout << "stop recoording" << endl;
            startRecording = 0;
            
            if(dataMatrix.empty()==true){
                cout << "You have not recorded anything!" << endl ;
            }else{
                for (int n = 0; n<=4; ++n){
                    if (boolVecModelsActive[n]==1){
                        modelsData[n].push_back(dataMatrix.clone());
                        modelsLabel[n].push_back(labelMatrix[n].clone());
                        labelMatrix[n].release();
                    }
                }
            }
            dataMatrix.release();//liberar memoria ya que se usa como aux para completar modelsData (vector donde esta toda la info)
            cout << "saved" << endl;
        }
        if (msg.getAddress()=="/wekinator/control/startRunning"){
            cout << "start running" << endl;
            startRunning = 1;
        }
        if (msg.getAddress()=="/wekinator/control/stopRunning"){
            cout << "stop running" << endl;
            startRunning = 0;
        }
        
        if (startRunning == 1){
            if (boolVecMethodsActive[0] == 1 ){ //Logistic Regression
                cout << "running with LR" << endl;
                float lrPrediction;
                for (int n = 0; n <= boolVecModelsActive.size()-1; n++){
                    if (boolVecModelsActiveRunning[n]==1){
                        lrPrediction = predictWithLR(lr[n],featuresViolinRT);
                        cout << lrPrediction << endl;
                    }
                }
            }
            if (boolVecMethodsActive[2] == 1){ //ANN
                cout << "Running with ANN " << endl;
                for (int n = 0; n <= boolVecModelsActive.size()-1; n++){
                    if (boolVecModelsActiveRunning[n] == 1){
                        senderOutput[n] = predictWithANN(ann[n], featuresViolinRT);
                        sendMsg.addFloatArg(senderOutput[n]);
                    }
                    else{
                        sendMsg.addFloatArg(0);
                    }
                    sender.sendMessage(sendMsg);
                    cout << "enviando modelo número " << n << "  por osc, valor: " <<senderOutput[n] << endl;
                    //                    cout << senderOutput.size() << endl;
                    //                    for (int i = 0; i <= 4; i++){
                    //                        cout << senderOutput[i] << endl;
                    //                    }
                }
            }
        }
        
        if ((startRecording == 1) && (msg.getAddress()=="/wek/inputs")){ //No guardamos vectores llenos de 0 sino solo cuando el pitch es detectado!!!!!SOLO GUARDA SI HAY PITCH Y ESTA EN MODO REC!!     //DEPRECATED (msg.getAddress()=="/wekinator/control/outputs")){ //data asíncrona!!!
            cout << "Grabando" << endl;
            saveDataRecorded(dataMatrix, labelMatrix, featuresViolinRT, isSoundGood,boolVecModelsActive);
        }
        
        if (msg.getAddress()=="/wekinator/control/enableModelRecording"){
            
            modelsActivated.clear();
            numModelsActivated = msg.getNumArgs();
            
            for (int numArg = 0; numArg <= numModelsActivated-1; ++numArg){ //Agafem el vector d'ints de models activats i mirem quins son
                modelsActivated.push_back(msg.getArgAsInt(numArg));
            }
            for (int numArg2 = 0; numArg2 <= numModelsActivated-1; ++numArg2){ //Del vector dactivats agafem un vector de bools i ho passem a 0 i 1
                boolVecModelsActive[modelsActivated[numArg2]-1]=true; // posem un 1 als q estan activats
            }
        }
        
        if (msg.getAddress()=="/wekinator/control/disableModelRecording"){ //FEm lo mateix pero pels desactivats
            
            modelsDesactivated.clear();
            
            for (int numArg = 0; numArg <= msg.getNumArgs()-1; ++numArg){
                modelsDesactivated.push_back(msg.getArgAsInt(numArg));
            }
            for (int numArg2 = 0; numArg2 <= msg.getNumArgs()-1; ++numArg2){
                boolVecModelsActive[modelsDesactivated[numArg2]-1] = false;
            }
        }
        
        if (msg.getAddress()=="/wekinator/control/enableModelRunning"){
            modelsActivatedRunning.clear();
            numModelsActivatedRunning = msg.getNumArgs();
            
            for(int numArgRun = 0; numArgRun <= numModelsActivatedRunning-1; ++ numArgRun){
                modelsActivatedRunning.push_back(msg.getArgAsInt(numArgRun));
            }
            for (int numArgRun2 = 0; numArgRun2 <= numModelsActivatedRunning-1; ++numArgRun2){
                boolVecModelsActiveRunning[modelsActivatedRunning[numArgRun2]-1]=true;
            }
        }
        
        if(msg.getAddress()=="/wekinator/control/disableModelRunning"){
            modelsDesactivatedRunning.clear();
            
            for(int numArgRun = 0; numArgRun <= msg.getNumArgs()-1; ++ numArgRun){
                modelsDesactivatedRunning.push_back(msg.getArgAsInt(numArgRun));
            }
            for(int numArgRun2 = 0; numArgRun2 <= msg.getNumArgs()-1; ++numArgRun2){
                boolVecModelsActiveRunning[modelsDesactivatedRunning[numArgRun2]-1]=false;
            }
        }
        
        if ((msg.getAddress()=="/wekinator/control/train") || (msg.getAddress()=="/wekinator/control/cancelTrain")){ //No podemos cancelar, así que cancelar y run activaran el train!
            
            if (boolLogisticRegression==true && boolSVM == true){
                cout << "ERROR!!: More than one method activated to train!! " << endl;
            }else{
                if (boolLogisticRegression == true){
                    
                    cout << "LogisticRegression TRAINING!" << endl;
                    createTrainData(modelsData, modelsLabel, boolVecModelsActive, boolVecMethodsActive, trainDataOfModels);
                    for (int n = 0; n<=4; ++n){
                        if(boolVecModelsActive[n]==1){
                            if(modelsData[n].size()<=1 ){
                                cout << "ERROR, No model selected or trainData created but cannot train because: data should have atleast 2 different classes, Model number -> " <<  n+1  << endl;
                            }else{
                                thetas[n] = trainWithLR(lr[n], trainDataOfModels[n], 0.01, 1000);
                                if (thetas[n].empty()){
                                    cout << "Train " << n+1 << " go wrong or not trained" << endl;
                                }else{
                                    cout << "training " << n << " ok" << endl;
                                    cout << thetas[n] << endl;
                                }
                            }
                        }
                    }
                }
                if (boolSVM == true){ // TODO: poder posar parámetres al SVM
                    cout << "SVM TRAINING!" << endl;
                    createTrainData(modelsData, modelsLabel, boolVecModelsActive, boolVecMethodsActive, trainDataOfModels);
                    for (int n = 0; n<=4; ++n){
                        if(boolVecModelsActive[n]==1){
                            if(modelsData[n].size()<=1 ){
                                cout << "ERROR, trainData created but cannot train because: data should have atleast 2 different classes, Model number -> " <<  n+1  << endl;
                            }else{
                                svmVectors[n] = trainWithSVM(svm, trainDataOfModels[n]); // TODO!!! fer que es pugui triar quins models entrenar
                                cout <<"Support Vectors: " << svmVectors[n] << endl;
                            }
                        }
                    }
                }
                if(boolANN == true){
                    cout << "ANN TRAINING" << endl;
                    createTrainData(modelsData, modelsLabel, boolVecModelsActive, boolVecMethodsActive, trainDataOfModels);
                    for (int n = 0; n <= boolVecModelsActive.size()-1; ++n){
                        if(boolVecModelsActive[n]==1){
                            if(modelsData[n].empty() ){
                                cout << "Data empty, Model number -> " <<  n+1  << endl;
                            }else{
                                trainWithANN(ann[n], trainDataOfModels[n]);
                                cout << "training Done!" << endl;
                            }
                        }
                    }
                }
            }
        }
        if(msg.getAddress()=="/wekinator/control/deleteAllExamples"){
            
            for (int n = 0; n < modelsData.size(); n++){
                modelsData[n].clear(); // Clear data saved
                modelsLabel[n].clear(); // Clear labels saved
                ann[n].release(); // Clear trainning set saved for ANN only!
            }
            cout << "All examples are delated" << endl;
        }
        if(msg.getAddress() == "/wekinator/control/wekSave"){
            
            vector<ofFile> vectorOfFiles = {firstModel,secondModel,thirdModel,fourthModel,fifthModel};
            saveDataRecordedToLocalMemory(vectorOfFiles, modelsData, modelsLabel);
            cout << "[Recorded Data] - Recorded Data storage process is finished." << endl;
            
            saveTrainedData(ann);
            cout << "[Ann Files Parameters] - Training Data storage process is finished." << endl;
            
        }
        
        if(msg.getAddress() == "/wekinator/control/wekLoad"){
            loadTrainingData(ann);
        }
    }
}

//--------------------------------------------------------------
void ofApp::draw(){
    
    ofDrawBitmapStringHighlight("Machine Learning OSC for ViolinRT", 50, 20);
    
    
    ofDrawBitmapStringHighlight("Model 1 Activated: " + ofToString(boolVecModelsActive[0], 7), 10, 50);
    ofDrawBitmapStringHighlight("Model 2 Activated: " + ofToString(boolVecModelsActive[1], 7), 10, 70);
    ofDrawBitmapStringHighlight("Model 3 Activated: " + ofToString(boolVecModelsActive[2], 7), 10, 90);
    ofDrawBitmapStringHighlight("Model 4 Activated: " + ofToString(boolVecModelsActive[3], 7), 10, 110);
    ofDrawBitmapStringHighlight("Model 5 Activated: " + ofToString(boolVecModelsActive[4], 7), 10, 130);
    
}
//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    
    if(key == 's'){
        vector<ofFile> vectorOfFiles = {firstModel,secondModel,thirdModel,fourthModel,fifthModel};
        saveDataRecordedToLocalMemory(vectorOfFiles, modelsData, modelsLabel);
        cout << "[Recorded Data] - Recorded Data storage process is finished." << endl;
        
        saveTrainedData(ann);
        cout << "[Ann Files Parameters] - Training Data storage process is finished." << endl;
    }
    
    if(key == 'r'){
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){
    
}
