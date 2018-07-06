#ifndef TLR_NEG_GT_LAYER_HPP_
#define TLR_NEG_GT_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class NegGTLayer : public Layer<Dtype>
{
 public:
  explicit NegGTLayer(const LayerParameter& param);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  //virtual int ExactNumBottomBlobs() const override;
  virtual int MinBottomBlobs() const override;
  virtual int MaxBottomBlobs() const override;
  virtual int MinTopBlobs() const override;
  virtual int MaxTopBlobs() const override;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom) override;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom) override;
 private:
  int batch_size_;
  int num_gt_;
};

// inline functions
template <typename Dtype>
inline NegGTLayer<Dtype>::NegGTLayer(const LayerParameter& param) 
  : Layer<Dtype>(param) {

}

template <typename Dtype>
inline const char* NegGTLayer<Dtype>::type() const {
  return "NegGT";
}

template <typename Dtype>
inline int NegGTLayer<Dtype>::MinBottomBlobs() const {
  return 0;
}

template <typename Dtype>
inline int NegGTLayer<Dtype>::MaxBottomBlobs() const {
  return 2;
}

template <typename Dtype>
inline int NegGTLayer<Dtype>::MinTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline int NegGTLayer<Dtype>::MaxTopBlobs() const {
  return 2;
}

template <typename Dtype>
inline void NegGTLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void NegGTLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

} // namespace bgm

#endif // !TLR_NEG_GT_LAYER_HPP_
