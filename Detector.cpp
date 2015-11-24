//
// Created by dneven on 11/20/15.
//

#include "Detector.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

Detector::Detector(const string& model_file,
                       const string& trained_file,
                       const int nClasses) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";

    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(nClasses, output_layer->channels())
        << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

std::vector<std::vector<RectWithScore> > Detector::nms(std::vector<std::vector<RectWithScore> > &list, float overlap, int offset)
{
    // As by http://vision.caltech.edu/~sbranson/code/
    std::vector<std::vector<RectWithScore> > bbsMap;
    for(int k = offset; k < list.size(); k++) {
        RectWithScore* boxes = &list[k][0];

        int numNMS = 0;
        int num = list[k].size();
        int i;
        // Greedy "non-maximal suppression"
        while(num) {
            // Greedily select the highest scoring bounding box
            int best = -1;
            float bestS = -10000000;
            for(i = 0; i < num; i++) {
                if(boxes[i].score > bestS) {
                    bestS = boxes[i].score;
                    best = i;
                }
            }
            //std::cout << "best score: " << bestS << std::endl;
            cv::Rect b = boxes[best].rect;
            RectWithScore tmp = boxes[0];
            boxes[0] = boxes[best];
            boxes[best] = tmp;
            boxes++;
            numNMS++;
            float A1 = b.width*b.height, A2, inter, inter_over_union;

            // Remove all bounding boxes where the percent area of overlap is greater than overlap
            int numGood = 0, x1, x2, y1, y2;
            for(i = 0; i < num-1; i++) {
                x1 = std::max(b.x, boxes[i].rect.x);
                y1 = std::max(b.y, boxes[i].rect.y);
                x2 = std::min(b.x+b.width,  boxes[i].rect.x+boxes[i].rect.width);
                y2 = std::min(b.y+b.height, boxes[i].rect.y+boxes[i].rect.height);
                A2 = boxes[i].rect.width*boxes[i].rect.height;
                inter = (float)((x2-x1)*(y2-y1));
                inter_over_union = inter / (A1+A2-inter);
                if(inter_over_union <= overlap) {
                    tmp = boxes[numGood];
                    boxes[numGood++] = boxes[i];
                    boxes[i] = tmp;
                }
            }
            num = numGood;
        }
        std::vector<RectWithScore> bbs(&list[k][0],&list[k][0] + numNMS);
        bbsMap.push_back(bbs);
    }
    return bbsMap;
}

static std::vector<std::vector<std::pair<int,float> > > threshold(const std::vector<std::vector<float> >& v, float t)
{
    std::vector<std::vector<std::pair<int,float> > > locationMap;
    for(int j = 0; j< v.size(); j++) {
        std::vector<std::pair<int, float> > locations;
        for (int i = 0; i < v[j].size(); i++)
            if (v[j][i] > t)
                locations.push_back(std::make_pair(i, v[j][i]));
        locationMap.push_back(locations);
    }
    return locationMap;
}

static std::vector<std::vector<RectWithScore > > listToBb(const std::vector<std::vector<std::pair<int,float> > > &list,int map_width,int win_size, int stride, double scale)
{
    std::vector<std::vector<RectWithScore > > bbsMap;
    for(int j = 0; j< list.size(); j++) {
        std::vector<RectWithScore> bbs;
        for (int i = 0; i < list[j].size(); i++) {
            cv::Rect rect((list[j][i].first % map_width) * stride * 1/scale,(list[j][i].first / map_width) * stride * 1/scale,win_size * 1/scale,win_size * 1/scale);
            float score = list[j][i].second;
            RectWithScore rws {rect, score};
            bbs.push_back(rws);
        }
        bbsMap.push_back(bbs);
    }
    return bbsMap;
}

std::vector<std::vector<RectWithScore> > Detector::Classify(const cv::Mat& img, int win_size, int win_stride, double score_threshold, double nms_overlap, double scale) {
    //std::cout << "Total images " << img.size() << std::endl;
    int map_width;
    std::vector<std::vector<float> > dmaps =  Predict(img,map_width);
    std::vector<std::vector<std::pair<int,float> > > locations = threshold(dmaps,score_threshold);
    std::vector<std::vector<RectWithScore> > bbs = listToBb(locations, map_width,win_size, win_stride, scale);
    return nms(bbs,nms_overlap,1);
}

std::vector<std::vector<float> > Detector::Predict(const cv::Mat& img, int& blob_width) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1,num_channels_,img.rows, img.cols);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->ForwardPrefilled();

    /* Copy the output layer to a std::vector */

    Blob<float>* output_layer = net_->output_blobs()[0];
    //std::cout << "Num of output channels " << output_layer->channels() << std::endl;
    //std::cout << "Width of blob" << output_layer->width() <<std::endl;
    blob_width = output_layer->width();
    //std::cout << "Height of blob" << output_layer->height() <<std::endl;
    std::vector<std::vector<float> > outputMaps;
    for(int i=0; i< output_layer->channels(); i++) {
        const float *begin = output_layer->cpu_data() + i * output_layer->width() * output_layer->height();
        const float *end = begin + output_layer->width() * output_layer->height();
        std::vector<float> o(begin, end);
        outputMaps.push_back(o);
    }
    return outputMaps;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
    vector<cv::Mat> channels;
    cv::split(img, channels);
    for (int j = 0; j < channels.size(); j++){
        channels[j].copyTo((*input_channels)[j]);
    }
}