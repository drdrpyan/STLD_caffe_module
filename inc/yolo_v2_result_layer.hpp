#ifndef TLR_BGM_YOLO_V2_RESULT_LAYER_HPP_
#define TLR_BGM_YOLO_V2_RESULT_LAYER_HPP_

#include "caffe/layer.hpp"

#include "yolo_v2_decoder.hpp"
#include "detection_nms.hpp"
#include "detection_encoder.hpp"

#include <memory>

namespace caffe
{

template <typename Dtype>
class YOLOV2ResultLayer : public Layer<Dtype>
{
 public:
  explicit YOLOV2ResultLayer(const LayerParameter& param);
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
  void InitDecoder(const YOLOV2ResultParameter& param);
  void InitDecoder(const YOLOV2ResultParameter& yolo_v2_result_param,
                   const SubwinOffsetParameter& subwin_offset_param);

  //cv::Size cell_size_;
  //int num_class_;
  //std::vector<cv::Rect_<Dtype> > anchor_;
  int num_detection_;
  bool do_nms_;
  //float nms_overlap_threshold_;

  float conf_threshold_;

  bool global_detection_;

  std::unique_ptr<bgm::YOLOV2Decoder<Dtype> > yolo_v2_decoder_;
  std::unique_ptr<bgm::DetectionEncoder<Dtype> > detection_encoder_;
};

// inline  functions
template <typename Dtype>
inline YOLOV2ResultLayer<Dtype>::YOLOV2ResultLayer(
    const LayerParameter& param) : Layer<Dtype>(param) {

}

template <typename Dtype>
inline const char* YOLOV2ResultLayer<Dtype>::type() const {
  return "YOLOV2Result";
}

template <typename Dtype>
inline int YOLOV2ResultLayer<Dtype>::ExactNumBottomBlobs() const {
  return 1;
}

template <typename Dtype>
inline int YOLOV2ResultLayer<Dtype>::ExactNumTopBlobs() const {
  return 3;
}

template <typename Dtype>
inline void YOLOV2ResultLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::vector<std::vector<bgm::Detection<Dtype> > > detection;
  yolo_v2_decoder_->Decode(*(bottom[0]), conf_threshold_, do_nms_,
                           &detection);

  detection_encoder_->Encode(detection, top);
}

template <typename Dtype>
inline void YOLOV2ResultLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void YOLOV2ResultLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void YOLOV2ResultLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

} // namespace caffe

#endif // !TLR_BGM_YOLO_V2_RESULT_LAYER_HPP_

