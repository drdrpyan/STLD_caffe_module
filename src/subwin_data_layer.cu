#include "subwin_data_layer.hpp"

namespace caffe
{

template <typename Dtype>
void SubwinDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_)
    prefetch_free_.push(prefetch_current_);
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  const Blob<Dtype>& src_data = prefetch_current_->data_;
  const Blob<Dtype>& src_label = prefetch_current_->label_;

  ReshapeTop(src_data, src_label, top);

  ForwardCroppedImg_gpu(src_data, top[0]);
  
  if (top.size() > 1)
    ForwardLabelBBox(src_label, top[1], top[2]);
  if (top.size() > 3)
    ForwardWholeImg(src_data, top[3]);
}

template <typename Dtype>
void SubwinDataLayer<Dtype>::ForwardCroppedImg_gpu(
    const Blob<Dtype>& src, Blob<Dtype>* dst) const {
  const Dtype* src_ptr = src.gpu_data();
  Dtype* dst_ptr = dst->mutable_gpu_data();

  for (int i = 0; i < win_offset_.size(); ++i) {
    for (int c = 0; c < src.channels(); ++c) {
      int src_offset = src.offset(0, c, win_offset_[i].y, win_offset_[i].x);
      const Dtype* src_iter = src_ptr + src_offset;

      int dst_offset = dst->offset(i, c);
      Dtype* dst_iter = dst_ptr + dst_offset;

      for (int h = dst->height(); h--; ) {
        caffe_copy<Dtype>(dst->width(), src_iter, dst_iter);
        src_iter += src.width();
        dst_iter += dst->width();
      }
    }
  }
}

template <typename Dtype>
void SubwinDataLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype> *> &bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(SubwinDataLayer);

} // namespace caffe