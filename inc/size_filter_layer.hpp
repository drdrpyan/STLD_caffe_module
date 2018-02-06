#ifndef TLR_SIZE_FILTER_LAYER_HPP_
#define TLR_SIZE_FILTER_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class SizeFilterLayer : public Layer<Dtype>
{
 public:
  SizeFilterLayer(const LayerParameter& param);
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
  int filter_axis_;
  Dtype min_;
  Dtype max_;
}; // class SizeFilterLayer

// inline functions
template <typename Dtype>
inline SizeFilterLayer<Dtype>::SizeFilterLayer(
    const LayerParameter& param)
  : Layer<Dtype>(param) {

}

template <typename Dtype>
inline const char* SizeFilterLayer<Dtype>::type() const {
  return "SizeFilter";
}

template <typename Dtype>
inline int SizeFilterLayer<Dtype>::ExactNumBottomBlobs() const {
  return 3;
}

template <typename Dtype>
inline int SizeFilterLayer<Dtype>::ExactNumTopBlobs() const {
  return 3;
}

template <typename Dtype>
inline void SizeFilterLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void SizeFilterLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if(propagate_down[0])
    bottom[0]->CopyFrom(*(top[0]), true);
  //if(propagate_down[1])
  //  bottom[1]->CopyFrom(*(top[1]), true);
}

template <typename Dtype>
inline void SizeFilterLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

} // namespace caffe

#endif // !TLR_SIZE_FILTER_LAYER_HPP_
