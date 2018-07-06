#ifndef TLR_DHA_RESULT_LAYER_HPP_
#define TLR_DHA_RESULT_LAYER_HPP_

#include "caffe/layer.hpp"

#include "detection_encoder.hpp"
#include "dha_decoder.hpp"
#include "dha_handler.hpp"

#include <opencv2/core.hpp>

namespace caffe
{

template <typename Dtype>
class DHAResultLayer : public Layer<Dtype>
{
  public:
  explicit DHAResultLayer(const LayerParameter& param);
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
  //void InitDecoder(const DHAResultParameter& param);
  void InitDecoder(const DHAResultParameter& dha_result_param,
                   const SubwinOffsetParameter& subwin_offset_param);

  cv::Size cell_size_;
  int num_class_;
  std::vector<cv::Rect_<Dtype> > anchor_;
  int num_detection_;
  bool do_nms_;
  float neg_threshold_;
  bool global_detection_;

  std::unique_ptr<bgm::DHADecoder<Dtype> > dha_decoder_;
  std::unique_ptr<bgm::DetectionEncoder<Dtype> > detection_encoder_;
};

// inline fucntions
template <typename Dtype>
inline DHAResultLayer<Dtype>::DHAResultLayer(
    const LayerParameter& param) : Layer<Dtype>(param) {

}

template <typename Dtype>
inline const char* DHAResultLayer<Dtype>::type() const {
  return "DHAResult";
}

template <typename Dtype>
inline int DHAResultLayer<Dtype>::ExactNumBottomBlobs() const {
  return 1;
}

template <typename Dtype>
inline int DHAResultLayer<Dtype>::ExactNumTopBlobs() const {
  return 3;
}

template <typename Dtype>
inline void DHAResultLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::vector<std::vector<bgm::Detection<Dtype> > > detection;
  dha_decoder_->Decode(*(bottom[0]), neg_threshold_, do_nms_,
                       &detection);

  detection_encoder_->Encode(detection, top);
}

template <typename Dtype>
inline void DHAResultLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void DHAResultLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void DHAResultLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

} // namespace caffe
#endif // !TLR_DHA_RESULT_LAYER_HPP_
