#include "size_rejection_layer.hpp"

#include <opencv2/core.hpp>

#include <numeric>
#include <limits>

namespace caffe
{

template <typename Dtype>
void SizeRejectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  const SizeRejectionParameter& param = this->layer_param_.size_rejection_param();

  const float FLOAT_MAX = std::numeric_limits<float>::max();
  const float FLOAT_MIN = -FLOAT_MAX;
  w_min_ = param.has_w_min() ? param.w_min() : FLOAT_MIN;
  w_max_ = param.has_w_max() ? param.w_max() : FLOAT_MAX;
  h_min_ = param.has_h_min() ? param.h_min() : FLOAT_MIN;
  h_max_ = param.has_h_max() ? param.h_max() : FLOAT_MAX;

  if (bottom.size() == 2) {
    anno_decoder_.reset(new bgm::AnnoDecoder<Dtype>);
    anno_encoder_.reset(new bgm::AnnoEncoder<Dtype>);
  }
  else {
    detection_decoder_.reset(new bgm::DetectionDecoder<Dtype>);
    detection_encoder_.reset(new bgm::DetectionEncoder<Dtype>);
  }
}

template <typename Dtype>
void SizeRejectionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 2) {
    std::vector<std::vector<int> > src_label;
    std::vector<std::vector<cv::Rect_<Dtype> > > src_bbox;
    anno_decoder_->Decode(bottom, &src_label, &src_bbox);

    std::vector<std::vector<int> > dst_label;
    std::vector<std::vector<cv::Rect_<Dtype> > > dst_bbox;
    for (int n = 0; n < src_label.size(); ++n) {
      for (int i = 0; i < src_label[n].size(); ++i) {
        int label = src_label[n][i];
        const cv::Rect_<Dtype>& bbox = src_bbox[n][i];
        if (label != LabelParameter::NONE ||
            label != LabelParameter::DUMMY_LABEL) {
          if (bbox.width >= w_min_ && bbox.width <= w_max_ &&
              bbox.height >= h_min_ && bbox.height <= h_max_) {
            dst_label[n].push_back(label);
            dst_bbox[n].push_back(bbox);
          }
        }
      }
    }

    std::vector<Blob<Dtype>*> top_blobs;
    top_blobs.assign(top.begin(), top.end());
    anno_encoder_->Encode(dst_label, dst_bbox, top_blobs);
  }
  else {
    std::vector<std::vector<bgm::Detection<Dtype> > > src_detection;
    detection_decoder_->Decode(bottom, &src_detection);

    std::vector<std::vector<bgm::Detection<Dtype> > > dst_detection;
    for (int n = 0; n < src_detection.size(); ++n) {
      for (int i = 0; i < src_detection[n].size(); ++i) {
        int label = src_detection[n][i].label;
        const cv::Rect_<Dtype>& bbox = src_detection[n][i].bbox;
        if (label != LabelParameter::NONE ||
            label != LabelParameter::DUMMY_LABEL) {
          if (bbox.width >= w_min_ && bbox.width <= w_max_ &&
              bbox.height >= h_min_ && bbox.height <= h_max_) {
            dst_detection[n].push_back(src_detection[n][i]);
          }
        }
      }
    }

    std::vector<Blob<Dtype>*> top_blobs;
    top_blobs.assign(top.begin(), top.end());
    detection_encoder_->Encode(dst_detection, top_blobs);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SizeRejectionLayer);
#endif

INSTANTIATE_CLASS(SizeRejectionLayer);
REGISTER_LAYER_CLASS(SizeRejection);

} // namespace caffe