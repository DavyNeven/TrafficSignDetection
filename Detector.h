//
// Created by dneven on 11/20/15.
//

#ifndef TESTCAFFE_DETECTOR_H
#define TESTCAFFE_DETECTOR_H


#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

struct RectWithScore
{
    cv::Rect rect;
    float score;
};

class Detector {
public:

    Detector(const string& model_file,
               const string& trained_file,
               const int nClasses);

    std::vector<std::vector<RectWithScore> > Classify(const cv::Mat& img, int win_size, int win_stride, float score_threshold, float nms_overlap);
private:

    std::vector<std::vector<float> > Predict(const cv::Mat& img);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);

private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    int nClasses;
};



#endif //TESTCAFFE_DETECTOR_H
