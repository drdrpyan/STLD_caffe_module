#ifndef TLR_CONTIAN_MINIBATCH_LAERY_HPP_
#define TLR_CONTIAN_MINIBATCH_LAERY_HPP_

#include "minibatch_data_layer.hpp"

namespace caffe
{

template <typename Dtype>
class ContainMinibatchLayer : public Layer<Dtype>
{
 public:
  explicit ContainMinibatchLayer(const LayerParameter& param);
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
};

// inline functions
template <typename Dtype>
inline ContainMinibatchLayer<Dtype>::ContainMinibatchLayer(
    const LayerParameter& param) : Layer<Dtype>(param) {

}

} // namespace caffe

#endif // !TLR_CONTIAN_MINIBATCH_LAERY_HPP_
