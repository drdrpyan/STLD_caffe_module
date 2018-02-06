#ifndef TLR_DETECTION_CHECK_LAYER_HPP_
#define TLR_DETECTION_CHECK_LAYER_HPP_

#include "caffe/layer.hpp"

#include <string>
#include <fstream>

#include "detection_decoder.hpp"
#include "img_decoder.hpp"
#include "detection_nms.hpp"

namespace caffe
{

template <typename Dtype>
class DetectionCheckLayer : public Layer<Dtype>
{
 public:
  explicit DetectionCheckLayer(const LayerParameter& param);
  virtual ~DetectionCheckLayer() override;
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int MinBottomBlobs() const override;
  virtual int MaxBottomBlobs() const override;

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
  void OpenOutStreams(const std::string& path);
  void CloseOutStreams();
  
  void DecodeDetection(
      const std::vector<Blob<Dtype>*>& bottom,
      std::vector<std::vector<bgm::Detection<Dtype> > >* detection);
  void DecodeImg(const std::vector<Blob<Dtype>*>& bottom,
                 std::vector<cv::Mat>* img);
  void DecodeGT(
      const std::vector<Blob<Dtype>*>& bottom,
      std::vector<std::vector<bgm::BoxAnnotation<Dtype> > >* gt);
  
  void LogDetection(
      const std::vector<std::vector<bgm::Detection<Dtype> > >& detection);
  void SaveImg(const std::string& path,
               const std::vector<cv::Mat>& img) const;
  void LogGT(
      const std::vector<std::vector<bgm::BoxAnnotation<Dtype> > >& gt);
  void LogEval(int img_cnt,
               const std::vector<int>& tp_idx,
               const std::vector<int>& fp_idx,
               const std::vector<int>& fn_idx);
  cv::Mat DrawResult(const std::vector<bgm::Detection<Dtype> >& detection,
                     const cv::Mat& img) const;
  cv::Mat DrawResultGT(const std::vector<bgm::Detection<Dtype> >& detection,
                       const std::vector<bgm::BoxAnnotation<Dtype> >& gt,
                       const std::vector<int>& tp_idx,
                       const std::vector<int>& fp_idx,
                       const std::vector<int>& fn_idx,
                       const cv::Mat& img) const;
  std::string GetImgName(int img_cnt) const;
  //void DrawAndSaveResult(
  //    const std::string& path,
  //    const std::vector<std::vector<bgm::Detection<Dtype> > >& detection,
  //    const std::vector<std::vector<bgm::BoxAnnotation<Dtype> > >& gt,
  //    const std::vector<cv::Mat>& img) const;

  bool eval_performance_;
  bool draw_;

  float conf_threshold_;
  float iou_threshold_;

  std::ofstream result_log_, gt_log_, eval_log_;
  std::string log_path_;

  bool detection_only_;

  std::unique_ptr<bgm::ImgDecoder<Dtype> > img_decoder_;
  std::unique_ptr<bgm::DetectionDecoder<Dtype> > detection_decoder_;
  std::unique_ptr<bgm::AnnoDecoder<Dtype> > anno_decoder_;

  bool do_nms_;
  std::unique_ptr<bgm::DetectionNMS<Dtype> > nms_;

  int img_cnt_;
  int tp_, fp_, fn_;
}; // class DetectionCheckLayer

// inline functions
template <typename Dtype>
DetectionCheckLayer<Dtype>::DetectionCheckLayer(
    const LayerParameter& param) : Layer<Dtype>(param) {

}

template <typename Dtype>
inline DetectionCheckLayer<Dtype>::~DetectionCheckLayer() {
  CloseOutStreams();
}

template <typename Dtype>
const char* DetectionCheckLayer<Dtype>::type() const {
  return "DetectionCheck";
}

template <typename Dtype>
inline int DetectionCheckLayer<Dtype>::MinBottomBlobs() const {
  return 4;
}

template <typename Dtype>
inline int DetectionCheckLayer<Dtype>::MaxBottomBlobs() const {
  return 6;
}

template <typename Dtype>
inline void DetectionCheckLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void DetectionCheckLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void DetectionCheckLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void DetectionCheckLayer<Dtype>::OpenOutStreams(
    const std::string& path) {
  result_log_.open(path + "/result.log");
  gt_log_.open(path + "/gt.log");
  eval_log_.open(path + "/eval.log");
}

template <typename Dtype>
inline void DetectionCheckLayer<Dtype>::CloseOutStreams() {
  result_log_.close();
  gt_log_.close();

  eval_log_ << "tp: " << tp_ << ", fp: " << fp_ << ", fn: " << fn_ << std::endl;
  eval_log_ << "precision: " << tp_ / static_cast<float>(tp_ + fp_);
  eval_log_ << ", recall: " << tp_ / static_cast<float>(tp_ + fn_) << std::endl;
  eval_log_.close();
}

template <typename Dtype>
inline void DetectionCheckLayer<Dtype>::DecodeImg(
    const std::vector<Blob<Dtype>*>& bottom, std::vector<cv::Mat>* img) {
  img_decoder_->Decode(*(bottom[3]), img);
}

template <typename Dtype>
inline void DetectionCheckLayer<Dtype>::DecodeGT(
    const std::vector<Blob<Dtype>*>& bottom,
    std::vector<std::vector<bgm::BoxAnnotation<Dtype> > >* gt) {
  std::vector<Blob<Dtype>*> gt_blobs(bottom.begin() + 4,
                                     bottom.begin() + 6); // 4, 5
  anno_decoder_->Decode(gt_blobs, gt);
}



} // namespace caffe
#endif // !TLR_DETECTION_CHECK_LAYER_HPP_
