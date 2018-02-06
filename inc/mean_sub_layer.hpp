#ifndef TLR_MEAN_SUB_LAYER_HPP_
#define TLR_MEAN_SUB_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class MeanSubLayer : public Layer<Dtype>
{
 public:
  explicit MeanSubLayer(const LayerParameter& param);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int ExactNumBottomBlobs() const override;
  virtual int ExactNumTopBlobs() const override;

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;

  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) override;
  virtual void Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) override;

 private:
  Blob<Dtype> data_mean_;
}; // class CheckLayer

// inline functions

template <typename Dtype>
inline MeanSubLayer<Dtype>::MeanSubLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {

}

template <typename Dtype>
inline const char* MeanSubLayer<Dtype>::type() const {
  return "MeanSub";
}

template <typename Dtype>
inline int MeanSubLayer<Dtype>::ExactNumBottomBlobs() const {
  return 1;
}

template <typename Dtype>
inline int MeanSubLayer<Dtype>::ExactNumTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline void MeanSubLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void MeanSubLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void MeanSubLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

} // namespace caffe

#endif // !TLR_MEAN_SUB_LAYER_HPP_