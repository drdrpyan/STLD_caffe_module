#ifndef TLR_DUMMY_LAYER_HPP_
#define TLR_DUMMY_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class DummyLayer : public Layer<Dtype>
{
  public:
    explicit DummyLayer(const LayerParameter& param);
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual void Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual const char* type() const override;

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
inline DummyLayer<Dtype>::DummyLayer(const LayerParameter& param) 
  : Layer<Dtype>(param) {

}

template <typename Dtype>
inline void DummyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
inline void DummyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<int> top_shape(4, 1);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
inline const char* DummyLayer<Dtype>::type() const {
  return "Dummy";
}

template <typename Dtype>
inline void DummyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  *(top[0]->mutable_cpu_data()) = 0;
}
    
template <typename Dtype>
inline void DummyLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void DummyLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void DummyLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

} // namespace caffe
#endif // !TLR_DUMMY_LAYER_HPP_