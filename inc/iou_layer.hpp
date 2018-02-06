#ifndef TLR_IOU_LAYER_HPP_
#define TLR_IOU_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class IOULayer : public Layer<Dtype>
{
 public:
  explicit IOULayer(const LayerParameter& param);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;

  virtual const char* type() const override;
  virtual int ExactNumBottomBlobs() const override;
  virtual int ExactNumTopBlobs() const override;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) override;

 private:
  Dtype ComputIOU(Dtype x, Dtype y, Dtype w, Dtype h,
                  Dtype gt_x, Dtype gt_y, 
                  Dtype gt_w, Dtype gt_h) const;
};

// inline functions
template <typename Dtype>
inline IOULayer<Dtype>::IOULayer(const LayerParameter& param)
  : Layer<Dtype>(param) {

}

template <typename Dtype>
inline const char* IOULayer<Dtype>::type() const {
  return "IOU";
}

template <typename Dtype>
inline int IOULayer<Dtype>::ExactNumBottomBlobs() const {
  return 3;
}

template <typename Dtype>
inline int IOULayer<Dtype>::ExactNumTopBlobs() const {
  return 1;
}

template <typename Dtype>
void IOULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {

}
}
#endif // !TLR_IOU_LAYER_HPP_
