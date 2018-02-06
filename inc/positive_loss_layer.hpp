#ifndef TLR_EMPTY_LOSS_LAYER_HPP_
#define TLR_EMPTY_LOSS_LAYER_HPP_

#include "caffe/layers/loss_layer.hpp"

#include "anno_decoder.hpp"

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"

#include <memory>

namespace caffe
{

template <typename Dtype>
class PositiveLossLayer : public LossLayer<Dtype>
{
 public:
  explicit PositiveLossLayer(const LayerParameter& param);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int ExactNumBottomBlobs() const override;
  virtual int ExactNumTopBlobs() const override;
  virtual int MinTopBlobs() const override;
  virtual int MaxTopBlobs() const override;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom) override;

 private:
  void CheckPositive(const Blob<Dtype>& label_gt, 
                     std::vector<bool>* empty_list) const;
  void MakePositiveGT(const std::vector<bool>& empty_list);
  void CalcPrRe(const Blob<Dtype>& input,
                const std::vector<bool>& gt,
                Dtype* precision, Dtype* recall) const;
  Dtype Sigmoid(Dtype value) const;

  shared_ptr<Layer<Dtype> > loss_layer_;
  std::vector<Blob<Dtype>*> loss_bottom_vec_, loss_top_vec_;

  Dtype threshold_;
  Blob<Dtype> positive_gt_;
};

// inline functions
template <typename Dtype>
inline PositiveLossLayer<Dtype>::PositiveLossLayer(const LayerParameter& param) 
  : LossLayer<Dtype>(param) {

}

template <typename Dtype>
inline const char* PositiveLossLayer<Dtype>::type() const {
  return "PositiveLoss";
}

template <typename Dtype>
inline int PositiveLossLayer<Dtype>::ExactNumBottomBlobs() const {
  return 2;
}

template <typename Dtype>
inline int PositiveLossLayer<Dtype>::ExactNumTopBlobs() const {
  return -1;
}

template <typename Dtype>
inline int PositiveLossLayer<Dtype>::MinTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline int PositiveLossLayer<Dtype>::MaxTopBlobs() const {
  return 3;
}

template <typename Dtype>
inline Dtype PositiveLossLayer<Dtype>::Sigmoid(Dtype value) const {
  return 1. / (1. + std::exp(-value));
}
} // namespace caffe

#endif // !TLR_EMPTY_LOSS_LAYER_HPP_