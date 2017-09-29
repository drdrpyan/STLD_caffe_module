#ifndef TLR_PADDING_LAYER_HPP_
#define TLR_PADDING_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class PaddingLayer : public Layer<Dtype>
{
 public:
  explicit PaddingLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual bool EqualNumBottomTopBlobs() const override;

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
  void MirrorPadding(const vector<Blob<Dtype>*>& bottom,
                     const vector<Blob<Dtype>*>& top) const;

  PaddingParameter::PaddingType TYPE_;
  const unsigned int PAD_UP_;
  const unsigned int PAD_DOWN_;
  const unsigned int PAD_LEFT_;
  const unsigned int PAD_RIGHT_;
};

// inline functions
template <typename Dtype>
inline void PaddingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
inline const char* PaddingLayer<Dtype>::type() const {
  return "Padding";
}

template <typename Dtype>
inline bool PaddingLayer<Dtype>::EqualNumBottomTopBlobs() const {
  return true;
}

template <typename Dtype>
inline void PaddingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void PaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down,
                                              const vector<Blob<Dtype>*>& bottom) {

}
template <typename Dtype>
inline void PaddingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down,
                                              const vector<Blob<Dtype>*>& bottom) {

}

} // namespace caffe

#endif // !TLR_PADDING_LAYER_HPP_