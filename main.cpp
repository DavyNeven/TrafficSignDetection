#include <iostream>
#include "Detector.h"

using namespace std;



int main() {
    string model_file = "/users/visics/dneven/ClionProjects/testCaffe/caffe_model/V10.prototxt";
    string trained_file = "/users/visics/dneven/ClionProjects/testCaffe/caffe_model/V10_iter_60000.caffemodel";

    Detector detector(model_file, trained_file, 9);

    vector<cv::Mat> images;

    cv::Mat img = cv::imread("/usr/data/dneven/datasets/GTSDB/data/Images/00000.ppm");
    cv::resize(img, img, cv::Size(680, 400));
    // Convert image to float
    img.convertTo(img, CV_32FC3, 1. / 255);

    std::vector<std::vector<RectWithScore > > outputMaps = detector.Classify(img,24, 4, 0.9, 0.3);
    for(int i=0; i< outputMaps.size();i++) {
        cout << "Total detections: " << outputMaps[i].size() << endl;
        for (int j = 0; j < outputMaps[i].size(); j++) {
            int x = outputMaps[i][j].rect.x;
            int y = outputMaps[i][j].rect.y;
            float score = outputMaps[i][j].score;
            cout << "X: " << x << " Y: " << y << " Score: " << score << endl;
            cv::rectangle(img, outputMaps[i][j].rect, cv::Scalar(255, 0, 0), 1, 8, 0);
        }
    }

    cv::imshow("image", img);
    cv::waitKey(0);
    return 0;
}