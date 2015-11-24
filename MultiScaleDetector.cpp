//
// Created by dneven on 11/24/15.
//

#include "MultiScaleDetector.h"

MultiScaleDetector::MultiScaleDetector(const string &model_file, const string &trained_file, const int nClasses,
                                       int win_size, int win_stride) {

    detector = new Detector(model_file, trained_file, nClasses);
    this->win_size = win_size;
    this->win_stride = win_stride;

}

std::vector<std::vector<RectWithScore> > MultiScaleDetector::detectMultiscale(const cv::Mat &img, float score_threshold,
                                                                              float nms_overlap,double *scales, int nScales) {
    cv::Mat copy = img;
    std::vector<std::vector<RectWithScore> > multiScaleDetections;
    for(int i = 0; i < nScales; i++)
    {
        cv::resize(img, copy,cv::Size(), scales[i],scales[i]);
        std::vector<std::vector<RectWithScore> > detections =
                detector->Classify(copy, win_size, win_stride, score_threshold, nms_overlap, scales[i]);
        if(multiScaleDetections.size() == 0)
            multiScaleDetections = detections;
        else
            for(int j = 0; j< detections.size(); j++)
                multiScaleDetections[j].insert(multiScaleDetections[j].end(), detections[j].begin(), detections[j].end());

    }
    return Detector::nms(multiScaleDetections,nms_overlap,0);
}
