#ifndef TLR_SOFTMAX_LOSS_LAYER_HPP_
#define TLR_SOFTMAX_LOSS_LAYER_HPP_

#include "caffe/layers/softmax_loss_layer.hpp"

#include <vector>

namespace caffe
{

template <typename Dtype>
class WeightedSoftmaxLossLayer : public SoftmaxWithLossLayer<Dtype>
{
 public:
  explicit WeightedSoftmaxLossLayer(const LayerParameter& param);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;

 protected:
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) override;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) override;
 private:
  std::vector<float> class_weight_;
   
};

// inline functions
template <typename Dtype>
inline const char* WeightedSoftmaxLossLayer<Dtype>::type() const {
  return "WeightedSoftmaxLoss";
}

template <typename Dtype>
void WeightedSoftmaxLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}
} // namespace caffe
#endif // !TLR_SOFTMAX_LOSS_LAYER_HPP_
