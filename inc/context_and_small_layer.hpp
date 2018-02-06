#ifndef TLR_CONTEXT_AND_SMALL_LAYER_HPP_
#define TLR_CONTEXT_AND_SMALL_LAYER_HPP_

#include "caffe/layer.hpp"

#include <opencv2/core.hpp>

namespace caffe
{

template <typename Dtype>
class ContextAndSmallLayer : public Layer<Dtype>
{
 public:
  explicit ContextAndSmallLayer(const LayerParameter& param);
  //virtual void LayerSetUp(
  //      const vector<Blob<Dtype>*>& bottom,
  //      const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;

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
  void BlobToCvMat(Blob<Dtype>& bottom_blob, 
                   std::vector<cv::Mat>* mat);
  void ExtractSmall(const std::vector<cv::Mat>& bottom_mat,
                    std::vector<cv::Mat>* small);

  cv::Rect small_area_;
}; // class ContextAndSmallLayer

// inline functions
template <typename Dtype>
inline ContextAndSmallLayer<Dtype>::ContextAndSmallLayer(
    const LayerParameter& param)
  : Layer<Dtype>(param),
    /*small_area_(48, 48, 32, 32)*/
    small_area_(38, 38, 52, 52) {
  // TODO : small_area_를 prototxt에서 정의하도록 caffe.proto에 추가할 것
}

template <typename Dtype>
inline const char* ContextAndSmallLayer<Dtype>::type() const {
  return "ContextAndSmall";
}

template <typename Dtype>
inline void ContextAndSmallLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void ContextAndSmallLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>  
inline void ContextAndSmallLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

} // namespace caffe

#endif // !TLR_CONTEXT_AND_SMALL_LAYER_HPP_
