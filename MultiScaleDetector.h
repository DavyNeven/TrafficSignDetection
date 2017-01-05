//
// Created by dneven on 11/24/15.
//

#ifndef TESTCAFFE_MULTISCALEDETECTOR_H
#define TESTCAFFE_MULTISCALEDETECTOR_H

#include "Detector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>


class MultiScaleDetector {


public:
    MultiScaleDetector(const string& model_file,
                       const string& trained_file,
                       const int nClasses,
                       int win_size, int win_stride);

    std::vector<RectWithScore> detectMultiscale(const cv::Mat& img, float score_threshold, float nms_overlap,
                                                                double *scales, int nScales);

protected:

    Detector *detector;
    int win_size;
    int win_stride;
};


#endif //TESTCAFFE_MULTISCALEDETECTOR_H
