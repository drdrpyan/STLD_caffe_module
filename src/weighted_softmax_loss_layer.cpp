#include "weighted_softmax_loss_layer.hpp"

namespace caffe
{

template <typename Dtype>
WeightedSoftmaxLossLayer<Dtype>::WeightedSoftmaxLossLayer(
    const LayerParameter& param) 
  : SoftmaxWithLossLayer<Dtype>(param) {

}

template <typename Dtype>
void WeightedSoftmaxLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  SoftmaxWithLossLayer<Dtype>::LayerSetUp(bottom, top);
  
  const WeightedSoftmaxLossParameter& param = layer_param_.weighted_softmax_loss_param();

  class_weight_.resize(param.class_weight().size());
  for (int i = 0; i < class_weight_.size(); ++i)
    class_weight_[i] = param.class_weight().Get(i);
}

template <typename Dtype>
void WeightedSoftmaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= class_weight_[label_value];
          ++count;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedSoftmaxLossLayer);
#endif

INSTANTIATE_CLASS(WeightedSoftmaxLossLayer);
REGISTER_LAYER_CLASS(WeightedSoftmaxLoss);

}