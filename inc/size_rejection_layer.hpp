#ifndef TLR_SIZE_REJECTION_LAYER_HPP_
#define TLR_SIZE_REJECTION_LAYER_HPP_

#include "caffe/layer.hpp"

#include "anno_decoder.hpp"
#include "anno_encoder.hpp"
#include "detection_decoder.hpp"
#include "detection_encoder.hpp"

#include <memory>

namespace caffe
{

template <typename Dtype>
class SizeRejectionLayer : public Layer<Dtype>
{
 public:
  explicit SizeRejectionLayer(const LayerParameter& param);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int MinBottomBlobs() const override;
  virtual int MaxBottomBlobs() const override;
  virtual int MinTopBlobs() const override;
  virtual int MaxTopBlobs() const override;
  virtual bool EqualNumBottomTopBlobs() const override;
  //virtual int ExactNumBottomBlobs() const override;
  //virtual int ExactNumTopBlobs() const override;

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
  float w_min_;
  float w_max_;
  float h_min_;
  float h_max_;

  std::unique_ptr<bgm::AnnoDecoder<Dtype> > anno_decoder_;
  std::unique_ptr<bgm::AnnoEncoder<Dtype> > anno_encoder_;
  std::unique_ptr<bgm::DetectionDecoder<Dtype> > detection_decoder_;
  std::unique_ptr<bgm::DetectionEncoder<Dtype> > detection_encoder_;
}; // class SizeRejectionLayer


// inline functions
template <typename Dtype>
inline SizeRejectionLayer<Dtype>::SizeRejectionLayer(
    const LayerParameter& param) 
  : Layer<Dtype>(param) {

}

template <typename Dtype>
inline void SizeRejectionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i)
    top[i]->ReshapeLike(*(bottom[i]));
}

template <typename Dtype>
inline const char* SizeRejectionLayer<Dtype>::type() const {
  return "SizeRejection";
}

//template <typename Dtype>
//inline int SizeRejectionLayer<Dtype>::ExactNumBottomBlobs() const {
//  return 2;
//}
//
//template <typename Dtype>
//inline int SizeRejectionLayer<Dtype>::ExactNumTopBlobs() const {
//  return 2;
//}

template <typename Dtype>
inline int SizeRejectionLayer<Dtype>::MinBottomBlobs() const {
  return 2;
}

template <typename Dtype>
inline int SizeRejectionLayer<Dtype>::MaxBottomBlobs() const {
  return 3;
}

template <typename Dtype>
inline int SizeRejectionLayer<Dtype>::MinTopBlobs() const {
  return 2;
}

template <typename Dtype>
inline int SizeRejectionLayer<Dtype>::MaxTopBlobs() const {
  return 3;
}

template <typename Dtype>
inline bool SizeRejectionLayer<Dtype>::EqualNumBottomTopBlobs() const {
  return true;
}

template <typename Dtype>
inline void SizeRejectionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void SizeRejectionLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void SizeRejectionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

} // namespace caffe
#endif // !TLR_SIZE_REJECTION_LAYER_HPP_
