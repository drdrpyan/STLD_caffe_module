#ifndef TLR_LABEL_DIFF_IGNORE_LAYER_HPP_
#define TLR_LABEL_DIFF_IGNORE_LAYER_HPP_

#include "boost/random.hpp"

#include "caffe/util/rng.hpp"
#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class LabelDiffIgnoreLayer : public Layer<Dtype>
{
  typedef boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<float> > RNG;
 public:
  explicit LabelDiffIgnoreLayer(const LayerParameter& param);
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
  void InitMaskGenerator();
  bool Reject(int label);
  
  std::vector<int> ignore_label_;
  std::vector<float> ignore_rate_;
  std::vector<RNG> mask_generator_;

  bool elem_wise_;
};

// inline functions
template <typename Dtype>
inline LabelDiffIgnoreLayer<Dtype>::LabelDiffIgnoreLayer(const LayerParameter& param) 
  : Layer<Dtype>(param) {

}

template <typename Dtype>
const char* LabelDiffIgnoreLayer<Dtype>::type() const {
  return "LabelDiffIgnore";
}

template <typename Dtype>
inline int LabelDiffIgnoreLayer<Dtype>::ExactNumBottomBlobs() const {
  return 2;
}

template <typename Dtype>
inline int LabelDiffIgnoreLayer<Dtype>::ExactNumTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline void LabelDiffIgnoreLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                     const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& label_probs = *(bottom[0]);
  Blob<Dtype>& label_probs_out = *(top[0]);
  label_probs_out.CopyFrom(label_probs);
}

template <typename Dtype>
inline void LabelDiffIgnoreLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                     const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& label_probs = *(bottom[0]);
  Blob<Dtype>& label_probs_out = *(top[0]);
  label_probs_out.CopyFrom(label_probs);
}

template <typename Dtype>
void LabelDiffIgnoreLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down,
                                               const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

} // namespace caffe
#endif // !TLR_LABEL_DIFF_IGNORE_LAYER_HPP_