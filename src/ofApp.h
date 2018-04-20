#pragma once

#include "ofMain.h"
#include "ofxOsc.h"
#include "logisticRegression.hpp"
#include "SVM.hpp"
#include "NeuralNetworks.hpp"
#include "utils.hpp"


class ofApp : public ofBaseApp{
    
public:
    void setup();
    void update();
    void draw();
    
    ofxOscSender sender;
    ofxOscReceiver receiver;
    ofxOscReceiver receiverRecord;
    
    ofFile firstModel;
    ofFile secondModel;
    ofFile thirdModel;
    ofFile fourthModel;
    ofFile fifthModel;
        
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
    
};
