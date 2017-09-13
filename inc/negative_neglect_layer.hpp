#ifndef TLR_NEGATIVE_NEGLECT_LAYER_HPP_
#define TLR_NEGATIVE_NEGLECT_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class NegativeNeglectLayer : public Layer<Dtype>
{
  public:
    NegativeNeglectLayer(const LayerParameter& param);
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
    int bbox_dim_;
}; // class ResolutionAugmentationLayer


// inline functions
template <typename Dtype>
inline NegativeNeglectLayer<Dtype>::NegativeNeglectLayer(const LayerParameter& param) 
  : Layer(param) {
}

template <typename Dtype>
inline void NegativeNeglectLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& predicted_bbox = *(bottom[0]);
  Blob<Dtype>& out = *(top[0]);

  out.ReshapeLike(predicted_bbox);
}

template <typename Dtype>
inline const char* NegativeNeglectLayer<Dtype>::type() const {
  std::vector<int>::const_iterator;
  return "NegativeNeglect";
}

template <typename Dtype>
int NegativeNeglectLayer<Dtype>::ExactNumBottomBlobs() const {
  return 2;
}

template <typename Dtype>
int NegativeNeglectLayer<Dtype>::ExactNumTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline void NegativeNeglectLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void NegativeNeglectLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0])
    bottom[0]->CopyFrom(*(top[0]), true);
}

template <typename Dtype>
inline void NegativeNeglectLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0])
    bottom[0]->CopyFrom(*(top[0]), true);
}


} // namespace caffe
#endif // !TLR_NEGATIVE_NEGLECT_LAYER_HPP_
