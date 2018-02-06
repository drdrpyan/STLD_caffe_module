#include "hard_negative_data_layer.hpp"

#include "caffe/util/math_functions.hpp"

#include <opencv2/core.hpp>

namespace caffe
{
template <typename Dtype>
void HardNegativeDataLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_)
    prefetch_free_.push(prefetch_current_);
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  Blob<Dtype>& top_data = *(top[0]);
  top_data.ReshapeLike(prefetch_current_->data_);

  if (top.size() > 1) {
    Blob<Dtype>& top_label = *(top[1]);
    std::vector<int> gt_shape(4, 1);
    gt_shape[0] = top_data.num();
    top_label.Reshape(gt_shape);
  }

  if(top.size() > 2) {
    Blob<Dtype>& top_bbox = *(top[2]);
    std::vector<int> gt_shape(4, 1);
    gt_shape[0] = top_data.num();
    gt_shape[1] = 4;
    top_bbox.Reshape(gt_shape);
  }

  if(top.size() > 3) {
    Blob<Dtype>& top_offset = *(top[3]);
    std::vector<int> gt_shape(4, 1);
    gt_shape[0] = top_data.num();
    gt_shape[1] = 4;
    top_offset.Reshape(gt_shape);
  }
}

template <typename Dtype>
void HardNegativeDataLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_)
    prefetch_free_.push(prefetch_current_);
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  std::vector<std::vector<Dtype> > gt_label;
  std::vector<std::vector<bgm::BBox<Dtype> > > gt_roi;
  BaseImgBBoxDataLayer<Dtype>::ParseLabelBBox(prefetch_current_->label_,
                                              &gt_label, &gt_roi);

  Blob<Dtype>& top_data = *(top[0]);
  top_data.ReshapeLike(prefetch_current_->data_);
  //top_data.ShareData(prefetch_current_->data_);
  top_data.CopyFrom(prefetch_current_->data_);

  if (top.size() > 1) {
    Blob<Dtype>& top_label = *(top[1]);
    //std::vector<int> gt_shape(1, 4);
    //gt_shape[0] = top_data.num();
    //top_label.Reshape(gt_shape);
    caffe::caffe_set(top_label.count(), static_cast<Dtype>(0), top_label.mutable_cpu_data());
  }

  if(top.size() > 2) {
    Blob<Dtype>& top_bbox = *(top[2]);
    //std::vector<int> gt_shape(1, 4);
    //gt_shape[0] = top_data.num();
    //gt_shape[1] = 4;
    //top_offset.Reshape(gt_shape);
    caffe::caffe_set(top_bbox.count(), static_cast<Dtype>(-1), top_bbox.mutable_cpu_data());
  }

  if(top.size() > 3) {
    Blob<Dtype>& top_offset = *(top[3]);
    //std::vector<int> gt_shape(1, 4);
    //gt_shape[0] = top_data.num();
    //gt_shape[1] = 4;
    //top_offset.Reshape(gt_shape);
    Dtype* iter = top_offset.mutable_cpu_data();
    for (int i = 0; i < top_offset.num(); ++i) {
      *iter++ = gt_roi[i][0].x_min();
      *iter++ = gt_roi[i][0].y_min();
      *iter++ = gt_roi[i][0].x_max() - gt_roi[i][0].x_min();
      *iter++ = gt_roi[i][0].y_max() - gt_roi[i][0].y_min();
    }
  }

  //// debug
  //int img_width = top_data.width();
  //int img_height = top_data.height();
  //int img_size = img_width * img_height;
  //Dtype* data_dst = top_data.mutable_cpu_data();
  //std::vector<cv::Mat> bgr(3);
  //bgr[0] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst);
  //bgr[1] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst + img_size);
  //bgr[2] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst + img_size * 2);
  //cv::Mat debug_data = cv::Mat(cv::Size(img_width, img_height), CV_32FC3);
  //cv::merge(bgr, debug_data);
  //debug_data.convertTo(debug_data, CV_8UC3);
}

#ifdef CPU_ONLY
STUB_GPU(HardNegativeDataLayer);
#endif

INSTANTIATE_CLASS(HardNegativeDataLayer);
REGISTER_LAYER_CLASS(HardNegativeData);

} // namespace caffe