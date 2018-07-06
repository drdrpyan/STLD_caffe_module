#ifndef TLR_PRECISION_RECALL_LAYER_HPP_
#define TLR_PRECISION_RECALL_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{
template <typename Dtype>
class PrecisionRecallLayer : public Layer<Dtype>
{
 public:
  explicit PrecisionRecallLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int ExactNumBottomBlobs() const override;
  virtual int ExactNumTopBlobs() const override;

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
  Dtype threshold_;
};

// inline functions
template<typename Dtype>
inline PrecisionRecallLayer<Dtype>::PrecisionRecallLayer(
    const LayerParameter& param) : Layer<Dtype>(param){

}

template<typename Dtype>
inline const char* PrecisionRecallLayer<Dtype>::type() const {
  return "PrecisionRecall";
}

template<typename Dtype>
inline int PrecisionRecallLayer<Dtype>::ExactNumBottomBlobs() const {
  return 2;
}

template<typename Dtype>
inline int PrecisionRecallLayer<Dtype>::ExactNumTopBlobs() const {
  return 4;
}

template<typename Dtype>
inline void PrecisionRecallLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template<typename Dtype>
inline void PrecisionRecallLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template<typename Dtype>
inline void PrecisionRecallLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}
}
#endif // !TLR_PRECISION_RECALL_LAYER_HPP_
