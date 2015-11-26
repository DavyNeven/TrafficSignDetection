//
// Created by dneven on 11/19/15.
//

#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__

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


class Classifier {
public:
    Classifier(const string& model_file,
               const string& trained_file,
                const int nClasses);

    std::vector< std::pair<int,float> > Classify(const std::vector<cv::Mat>& img);

private:

    std::vector< std::vector<float> > Predict(const std::vector<cv::Mat>& img, int nImages);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels, int nImages);

    void Preprocess(const std::vector<cv::Mat>& img,
                    std::vector<cv::Mat>* input_channels, int nImages);

private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
};

#endif /* __CLASSIFIER_H__ */
