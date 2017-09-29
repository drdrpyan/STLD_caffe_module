#ifndef TLR_SIZE_UNIT_LAYER_HPP_
#define TLR_SIZE_UNIT_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{
template <typename Dtype>
class SizeUnitLayer : public Layer<Dtype>
{
 public:
  explicit SizeUnitLayer(const LayerParameter& param);
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

  /**
  This layer handles input data. It does not backprogate.
  */
  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) override;
  /**
  This layer handles input data. It does not backprogate.
  */
  virtual void Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) override;

 private:
  const float UNIT_;
};

// inline functions
template <typename Dtype>
inline SizeUnitLayer<Dtype>::SizeUnitLayer(const LayerParameter& param) 
  : Layer<Dtype>(param),
    UNIT_(param.size_unit_param().unit()) {

}

template <typename Dtype>
inline void SizeUnitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
inline void SizeUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*(bottom[0]));
}

template <typename Dtype>
inline const char* SizeUnitLayer<Dtype>::type() const {
  return "SizeUnit";
}

template <typename Dtype>
inline int SizeUnitLayer<Dtype>::ExactNumBottomBlobs() const {
  return 1;
}

template <typename Dtype>
inline int SizeUnitLayer<Dtype>::ExactNumTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline void SizeUnitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}


template <typename Dtype>
inline void SizeUnitLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void SizeUnitLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

}
#endif // !TLR_SIZE_UNIT_LAYER_HPP_
