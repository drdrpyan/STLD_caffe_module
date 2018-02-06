#ifndef TLR_ANNO_TO_SEG_LAYER_HPP_
#define TLR_ANNO_TO_SEG_LAYER_HPP_

#include "caffe/layer.hpp"

#include <opencv2/core.hpp>

namespace caffe
{

template <typename Dtype>
class AnnoToSegLayer : public Layer<Dtype>
{
 public:
  explicit AnnoToSegLayer(const LayerParameter& param);
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

  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) override;
  virtual void Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) override;

 private:
  void ParseGT(const Blob<Dtype>& label_blob,
               const Blob<Dtype>& bbox_blob,
               std::vector<std::vector<int> >* label,
               std::vector<std::vector<cv::Rect_<Dtype> > >* bbox) const;
  //void MakeSegMap()

  bool objectness_;
  int num_label_;

  bool bbox_normalized_;
  cv::Size in_size_;
  cv::Size out_size_;
};

// inline functions
template <typename Dtype>
inline AnnoToSegLayer<Dtype>::AnnoToSegLayer(const LayerParameter& param) 
  : Layer<Dtype>(param) {
  
}

template <typename Dtype>
inline const char* AnnoToSegLayer<Dtype>::type() const {
  return "AnnoToSeg";
}

template <typename Dtype>
inline int AnnoToSegLayer<Dtype>::ExactNumBottomBlobs() const {
  return 2;
}

template <typename Dtype>
inline int AnnoToSegLayer<Dtype>::ExactNumTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline void AnnoToSegLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void AnnoToSegLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void AnnoToSegLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}
} // namespace caffe

#endif // !TLR_ANNO_TO_SEG_LAYER_HPP_
