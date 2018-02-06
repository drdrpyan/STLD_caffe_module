#ifndef TLR_BALANCED_LABEL_DIFF_IGNORE_LAYER_HPP_
#define TLR_BALANCED_LABEL_DIFF_IGNORE_LAYER_HPP_

#include "caffe/layer.hpp"

#include <vector>

namespace caffe
{

template <typename Dtype>
class BalancedLabelDiffIgnoreLayer : public Layer<Dtype>
{
 public:
  explicit BalancedLabelDiffIgnoreLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int ExactNumBottomBlobs() const override;
  virtual int ExactNumTopBlobs() const override;

 protected:
 private:
  const Dtype NEG_RATIO_;
  std::vector<Dtype> pos_;
  std::vector<Dtype> neg_;
  
  
}; // class BalancedLabelDiffIgnoreLayer

} // namespace caffe

#endif // !TLR_BALANCED_LABEL_DIFF_IGNORE_LAYER_HPP_
