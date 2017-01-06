#include <iostream>
#include "MultiScaleDetector.h"
#include "Classifier.h"

using namespace std;

int nScales = 3;
double scales[] = {0.8, 0.4, 0.2};
int w = 640;
int h = 480;
double score_threshold = 0.99;
double nms_overlap = 0.3;

//Model parameters
int downsample = 4;
int normsize = 128/4;

vector<cv::Mat> images;
vector<cv::Mat> labels;

string det_model_file = "/users/visics/dneven/Devel/C++/TrafficSignDetection/caffe_model/det.prototxt";
string det_caffe_file = "/users/visics/dneven/Devel/C++/TrafficSignDetection/caffe_model/det.caffemodel";

string rec_model_file = "/users/visics/dneven/Devel/C++/TrafficSignDetection/caffe_model/clas.prototxt";
string rec_caffe_file = "/users/visics/dneven/Devel/C++/TrafficSignDetection/caffe_model/clas.caffemodel";

string imagesBasePath = "/esat/larimar/dneven/datasets/GTSDB/FullIJCNN2013/";
string labelBasePath = "/esat/larimar/dneven/datasets/GTSRB/classImages/";

void loadImages()
{
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

    loadImages();
    cv::VideoCapture cap;
    initWebcam(cap, 0, w, h);
    cv::Mat webcamImage;
   

    //Detector detector(model_file, trained_file, 9);
    MultiScaleDetector detector(det_model_file, det_caffe_file, 2, normsize, downsample);
    Classifier classifier(rec_model_file, rec_caffe_file, 43);

    clock_t start, end;
    start = clock();
    //for(int k = 0; k< images.size(); k++) {
    while(true) {
	    cap >> webcamImage;
        webcamImage.convertTo(webcamImage, CV_32FC3, 1. / 255);
	    vector<RectWithScore> bbs = detector.detectMultiscale(webcamImage, score_threshold, nms_overlap, scales, nScales);
        vector<cv::Mat> patches;
        for (int i = 0; i < bbs.size(); i++){
                cv::Rect rect = bbs[i].rect;
                rect.width = min(w, rect.width);
                rect.height = min(h, rect.height);
                rect.x = max(0, rect.x);
                rect.y = max(0, rect.y);
                cout << bbs[i].rect << endl;
                cv::Mat patch = webcamImage(bbs[i].rect);
                cv::resize(patch, patch, cv::Size(48, 48));
                patches.push_back(patch);
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
        for (int i = 0; i < bbs.size(); i++) {
            //cout << "Class " << i << endl;
                cv::rectangle(webcamImage, bbs[i].rect, cv::Scalar(255,0,0), 1, 8, 0);
                cv::Mat small = labels[classifications[count].first];
		        if(classifications[count].second > 0.8)
			        small.copyTo(webcamImage(cv::Rect(max(0,bbs[i].rect.tl().x-30), max(0,bbs[i].rect.tl().y-30),small.cols, small.rows)));
                cout << "score detection: " << bbs[i].score << endl;
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



