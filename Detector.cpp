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
    CHECK_EQ(net_->num_outputs(), 2) << "Network should have two outputs.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3)
    << "Input layer should have 3 channels.";

    Blob<float>* score_layer = net_->output_blobs()[1];
    CHECK_EQ(2, score_layer->channels())
        << "Dimension of score output is not equal to 2.";

    Blob<float>* bbox_layer = net_->output_blobs()[0];
    CHECK_EQ(4, bbox_layer->channels())
        << "Dimension of bbox output is not equal to 4.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

std::vector<RectWithScore> Detector::nms(std::vector<RectWithScore> &list, float overlap)
{
    RectWithScore* boxes = &list[0];

    int numNMS = 0;
    int num = list.size();
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
    std::vector<RectWithScore> bbs(&list[0],&list[0] + numNMS);

    return bbs;
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

/*static std::vector<std::vector<RectWithScore > > listToBb(const std::vector<std::vector<std::pair<int,float> > > &list,int map_width,int win_size, int stride, double scale)
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
}*/

static std::vector<RectWithScore> listToBb(const std::vector<std::vector<float> > &list,int map_width,int win_size, int stride, double scale, double score_threshold)
{
    std::vector<RectWithScore> bbs;
    for(int i = 0; i< list[0].size(); i++){
        float score = list[0][i];
        //std::cout << score << std::endl;
        if(score > score_threshold){
            int x =  (i%map_width)*stride;
            int y = (i/map_width)*stride;
            int x1 = std::max(0.0, (x - list[1][i]*win_size)*1/scale);
            int y1 = std::max(0.0, (y - list[2][i]*win_size)*1/scale);
            int x2 = (x - list[3][i]*win_size)*1/scale;
            int y2 = (y - list[4][i]*win_size)*1/scale;
            int w = x2 - x1;
            int h = y2 - y1;
            if(w > 0 && h > 0) {
                cv::Rect rect(x1, y1, w, h);
                RectWithScore rws{rect, score};
                bbs.push_back(rws);
            }
        }
    }
    return bbs;
}



std::vector<RectWithScore> Detector::Classify(const cv::Mat& img, int win_size, int win_stride, double score_threshold, double nms_overlap, double scale) {
    //std::cout << "Total images " << img.size() << std::endl;
    int map_width;
    std::vector<std::vector<float> > outputMaps =  Predict(img,map_width);
    std::vector<RectWithScore> bbs = listToBb(outputMaps, map_width,win_size, win_stride, scale, score_threshold);
    return nms(bbs,nms_overlap);
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

    Blob<float>* score_map = net_->output_blobs()[1];
    Blob<float>* bbox_map = net_->output_blobs()[0];

    blob_width = score_map->width();
    //std::cout << "Height of blob" << output_layer->height() <<std::endl;
    std::vector<std::vector<float> > outputMaps;

    // Read one score map
    const float *s_begin = score_map->cpu_data() + score_map->width()*score_map->height();
    const float *s_end = s_begin + score_map->width()*score_map->height();
    std::vector<float> o(s_begin, s_end);
    outputMaps.push_back(o);

    // Read 4 bbox maps
    for(int i=0; i< bbox_map->channels(); i++) {
        const float *begin = bbox_map->cpu_data() + i * bbox_map->width() * bbox_map->height();
        const float *end = begin + bbox_map->width() * bbox_map->height();
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