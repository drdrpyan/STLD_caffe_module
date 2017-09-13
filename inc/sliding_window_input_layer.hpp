#ifndef TLR_SLIDING_WIDNOW_INPUT_LAYER_HPP_
#define TLR_SLIDING_WIDNOW_INPUT_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class SlidingWindowInputLayer : public Layer<Dtype>
{
 public:
  SlidingWindowInputLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int MinTopBlobs() const override;
  virtual int MaxTopBlobs() const override;
  virtual int ExactNumBottomBlobs() const override;

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
  void UpdateOffsets(int num_img, int input_width, int input_height);
  void ComputeOffsets();

  const int WINDOW_WIDTH_;
  const int WINDOW_HEIGHT_;
  const int HORIZONTAL_STRIDE_;
  const int VERTICAL_STRIDE_;
  const bool WIN_NORMALIZATION_;

  int num_img_;
  int input_width_;
  int input_height_;
  Blob<Dtype> offsets_;
};

// inline functions

template <typename Dtype>
inline void SlidingWindowInputLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
inline const char* SlidingWindowInputLayer<Dtype>::type() const {
  return "SlidingWindowInput";
}

template <typename Dtype>
inline int SlidingWindowInputLayer<Dtype>::MinTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline int SlidingWindowInputLayer<Dtype>::MaxTopBlobs() const {
  return 2;
}

template <typename Dtype>
inline int SlidingWindowInputLayer<Dtype>::ExactNumBottomBlobs() const {
  return 1;
}

template <typename Dtype>
inline void SlidingWindowInputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //const Blob<Dtype>& input_img = *(bottom[0]);
  //Blob<Dtype>& output_img = *(top[0]);
  //output_img.CopyFrom(input_img);
  top[0]->CopyFrom(*(bottom[0]));

  if (top.size() > 1)
    top[1]->CopyFrom(offsets_);
}

template <typename Dtype>
inline void SlidingWindowInputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void SlidingWindowInputLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, 
    const vector<Blob<Dtype>*>& bottom) {
}

template <typename Dtype>
inline void SlidingWindowInputLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}



} // namespace caffe

#endif // !TLR_SLIDING_WIDNOW_INPUT_LAYER_HPP_
