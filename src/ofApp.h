#pragma once

#include "ofMain.h"
#include "ofxOsc.h"
#include "logisticRegression.hpp"
#include "SVM.hpp"
#include "NeuralNetworks.hpp"
#include "utils.hpp"
#include <string>
#include <opencv2/opencv.hpp>
//#include "ofBaseApp.h"

//#ifndef OFBASEAPP_H
//#define OFBASEAPP_H

class ofApp : public ofBaseApp{
    
public:
    void setup();
    void update();
    void draw();
    
    ofxOscSender sender;
    ofxOscReceiver receiver;
    ofxOscReceiver receiverRecord;
    
    ofFile firstModel,secondModel,thirdModel,fourthModel,fifthModel;
    ofFile firstModelWeights,secondModelWeights,thirdModelWeights,fourthModelWeights,fifthModelWeights;
        
    float peak;
    float freq;
    float label;
    
    
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void mouseEntered(int x, int y);
    void mouseExited(int x, int y);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
    
    //bool sortColorFunction (ofColor i,ofColor j);
    vector<ofImage>loadedImages;
    vector<ofImage>processedImages;
    string originalFileExtension;
    
};
