#include <iostream>
#include "MultiScaleDetector.h"
#include <time.h>

using namespace std;






int main() {
    string model_file = "/users/visics/dneven/ClionProjects/testCaffe/caffe_model/V10.prototxt";
    string trained_file = "/users/visics/dneven/ClionProjects/testCaffe/caffe_model/V10_iter_59000.caffemodel";

    //Detector detector(model_file, trained_file, 9);
    MultiScaleDetector detector(model_file, trained_file, 9, 24, 4);
    vector<cv::Mat> images;
    for(int i = 0; i< 100; i++){
        char buffer[10];
        sprintf(buffer, "%05d", i);
        string path = "/users/visics/dneven/datasets/GTSDB/data/Images/" + string(buffer) + ".ppm";
        cv::Mat img = cv::imread(path);
        cv::resize(img, img, cv::Size(680, 400));
        img.convertTo(img, CV_32FC3, 1. / 255);
        images.push_back(img);
    }

    vector<cv::Scalar> colormap;
    colormap.push_back(cv::Scalar(0,100,200));
    colormap.push_back(cv::Scalar(255,0,0));
    colormap.push_back(cv::Scalar(0,255,0));
    colormap.push_back(cv::Scalar(255,255,0));
    colormap.push_back(cv::Scalar(0,0,255));
    colormap.push_back(cv::Scalar(255,0,255));
    colormap.push_back(cv::Scalar(0,255,255));
    colormap.push_back(cv::Scalar(255,255,255));
    colormap.push_back(cv::Scalar(100,200,50));

    double scales[] = {1.4,1.2 ,1, 0.8, 0.6, 0.4, 0.2};
    int nScales = 7;

    clock_t start, end;
    start = clock();
    for(int k = 0; k< images.size(); k++) {
        //Convert image to float
        cv::Mat img = images[k];
        cv::Mat imgNMS = images[k].clone();

        vector<vector<RectWithScore> > outputMaps = detector.detectMultiscale(img, 0.9, 0.1, scales, nScales);

        for (int i = 0; i < outputMaps.size(); i++) {
            //cout << "Total detections: " << outputMaps[i].size() << endl;
            for (int j = 0; j < outputMaps[i].size(); j++) {
                int x = outputMaps[i][j].rect.x;
                int y = outputMaps[i][j].rect.y;
                float score = outputMaps[i][j].score;
                //cout << "X: " << x << " Y: " << y << " Score: " << score << endl;
                cv::rectangle(img, outputMaps[i][j].rect, colormap[i], 1, 8, 0);
            }
        }
        cv::imshow("image", images[k]);
        cv::waitKey(0);
    }
    end = clock();
    cout << "Time required for execution: "
    << (double)(end-start)/CLOCKS_PER_SEC
    << " seconds." << "\n\n";
    cv::waitKey(0);
    return 0;
}