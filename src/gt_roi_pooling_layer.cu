#include "gt_roi_pooling_layer.hpp"

#include "caffe/util/math_functions.hpp"

namespace caffe 
{

template <typename Dtype>
void GTROIPoolingLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<std::vector<int> > roi_label;
  std::vector<std::vector<cv::Rect_<Dtype> > > roi_bbox;
  PickRandomROI(bottom, &roi_label, &roi_bbox);

  PoolROI_gpu(*(bottom[0]), top[0]);
  MakeGTTop(roi_label, roi_bbox, top[1], top[2]);
}

template <typename Dtype>
void GTROIPoolingLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0])
    return;

  Dtype* bot_diff_ptr = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set<Dtype>(bottom[0]->count(), static_cast<Dtype>(0),
                       bot_diff_ptr);

  const Dtype* top_diff_ptr = top[0]->gpu_diff();

  for (auto iter = roi_relation_.cbegin();
       iter != roi_relation_.cend(); ++iter) {

    for (int c = 0; c < bottom[0]->channels(); ++c) {
      int bot_offset = bottom[0]->offset(iter->bot_idx, c, iter->offset_y, iter->offset_x);
      Dtype* bot_diff_iter = bot_diff_ptr + bot_offset;

      const Dtype* top_diff_iter = top_diff_ptr + top[0]->offset(iter->top_idx, c);

      for (int h = roi_size_.height; h--;) {
        caffe_gpu_axpy(roi_size_.width, static_cast<Dtype>(1),
                       top_diff_iter, bot_diff_iter);
        bot_diff_iter += bottom[0]->width();
        top_diff_iter += roi_size_.width;
      }
    }
  }
}

template <typename Dtype>
void GTROIPoolingLayer<Dtype>::PoolROI_gpu(
    const Blob<Dtype>& bottom, Blob<Dtype>* top) const {
  const Dtype* bot_ptr = bottom.gpu_data();
  Dtype* top_ptr = top->mutable_gpu_data();

  for (auto iter = roi_relation_.cbegin(); iter != roi_relation_.cend();
       ++iter) {
    for (int c = 0; c < bottom.channels(); ++c) {
      int bot_offset = bottom.offset(iter->bot_idx, c, iter->offset_y, iter->offset_x);
      const Dtype* bot_iter = bot_ptr + bot_offset;
      Dtype* top_iter = top_ptr + top->offset(iter->top_idx);

      for (int h = roi_size_.height; h--;) {
        caffe_copy(roi_size_.width, bot_iter, top_iter);
        bot_iter += bottom.width();
        top_iter += roi_size_.width;
      }
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(GTROIPoolingLayer);

}