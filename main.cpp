#include <iostream>
#include "MultiScaleDetector.h"
#include "Classifier.h"
#include <time.h>

using namespace std;


double scales[] = {0.8,0.6,0.4,0.2};
int nScales = 4;
vector<cv::Scalar> colormap;
vector<cv::Mat> images;
vector<cv::Mat> labels;

string det_model_file = "/users/visics/dneven/Devel/C++/TrafficSignDetection/caffe_model/det.prototxt";
string det_caffe_file = "/users/visics/dneven/Devel/C++/TrafficSignDetection/caffe_model/det.caffemodel";

string rec_model_file = "/users/visics/dneven/Devel/C++/TrafficSignDetection/caffe_model/Classifier/MC.prototxt";
string rec_caffe_file = "/users/visics/dneven/Devel/C++/TrafficSignDetection/caffe_model/classifier.caffemodel";

string imagesBasePath = "/esat/larimar/dneven/datasets/GTSDB/FullIJCNN2013/";
string labelBasePath = "/esat/larimar/dneven/datasets/GTSRB/classImages/";

void createColorMap()
{
    colormap.push_back(cv::Scalar(0,100,200));
    colormap.push_back(cv::Scalar(255,0,0));
    colormap.push_back(cv::Scalar(0,255,0));
    colormap.push_back(cv::Scalar(255,255,0));
    colormap.push_back(cv::Scalar(0,0,255));
    colormap.push_back(cv::Scalar(255,0,255));
    colormap.push_back(cv::Scalar(0,255,255));
    colormap.push_back(cv::Scalar(255,255,255));
    colormap.push_back(cv::Scalar(100,200,50));
}

void loadImages()
{
    /*for(int i = 0; i< 100; i++){
        char buffer[10];
        sprintf(buffer, "%05d", i);
        string path = imagesBasePath + string(buffer) + ".ppm";
        cv::Mat img = cv::imread(path);
        cv::resize(img, img, cv::Size(640, 400));
        img.convertTo(img, CV_32FC3, 1. / 255);
        images.push_back(img);
    }*/
    for(int i = 0; i< 43; i++){
        char buffer[10];
        sprintf(buffer, "%05d", i);
        string path = labelBasePath + string(buffer) + ".png";
        cv::Mat img = cv::imread(path);
        cv::resize(img, img, cv::Size(30, 30));
        img.convertTo(img, CV_32FC3, 1. / 255);
        labels.push_back(img);
    }
}

void init()
{
    createColorMap();
    loadImages();
}

void initWebcam(cv::VideoCapture &cap, int cameraNumber,int width, int height)
{
    cap.open(cameraNumber);
    if(!cap.isOpened())
    {
        cerr << "ERROR: could not acces this camera or video!" << endl;
        exit(1);
    }
    //Try to set camera resolution

    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
}

int main() {

    init();
    cv::VideoCapture cap;
    initWebcam(cap, 0, 640, 480);
    cv::Mat webcamImage;
   

    //Detector detector(model_file, trained_file, 9);
    MultiScaleDetector detector(det_model_file, det_caffe_file, 9, 20, 4);
    Classifier classifier(rec_model_file, rec_caffe_file, 43);

    clock_t start, end;
    start = clock();
    //for(int k = 0; k< images.size(); k++) {
    while(true) {
	cap >> webcamImage;
        webcamImage.convertTo(webcamImage, CV_32FC3, 1. / 255);
	vector<vector<RectWithScore> > outputMaps = detector.detectMultiscale(webcamImage, 0.8, 0.2, scales, nScales);
        vector<cv::Mat> patches;
        for (int i = 0; i < outputMaps.size(); i++) {
            for (int j = 0; j < outputMaps[i].size(); j++) {
                cv::Mat patch = webcamImage(outputMaps[i][j].rect);
                cv::resize(patch, patch, cv::Size(48, 48));
                patches.push_back(patch);
            }
        }
        vector<pair<int,float> > classifications;
        if(patches.size() > 0)
        {
            classifications = classifier.Classify(patches);
            for(int i=0; i<classifications.size(); i++)
            {
                cout << "classified as class: " << classifications[i].first << " with score: " << classifications[i].second << endl;
            }
        }
        int count = 0;
        for (int i = 0; i < outputMaps.size(); i++) {
            //cout << "Class " << i << endl;
            for (int j = 0; j < outputMaps[i].size(); j++) {
                cv::rectangle(webcamImage, outputMaps[i][j].rect, colormap[i], 1, 8, 0);
                cv::Mat small = labels[classifications[count].first];
		if(classifications[count].second > 0.8)                
			small.copyTo(webcamImage(cv::Rect(max(0,outputMaps[i][j].rect.tl().x-30), max(0,outputMaps[i][j].rect.tl().y-30),small.cols, small.rows)));
                cout << "score detection: " << outputMaps[i][j].score << endl;
                count++;
            }

        }
        cv::imshow("image", webcamImage);
        cv::waitKey(10);
    }
    end = clock();
    cout << "Time required for execution: "
    << (double)(end-start)/CLOCKS_PER_SEC
    << " seconds." << "\n\n";
    cv::waitKey(0);
    return 0;
}



