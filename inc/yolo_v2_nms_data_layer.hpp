#ifndef TLR_YOLO_V2_NMS_DATA_LAYER_HPP_
#define TLR_YOLO_V2_NMS_DATA_LAYER_HPP_

#include "caffe/layer.hpp"

#include "anno_decoder.hpp"
#include "yolo_v2_handler.hpp"

#include <memory>

namespace caffe
{

template <typename Dtype>
class YOLOV2NMSDataLayer : public Layer<Dtype>
{
 public:
  explicit YOLOV2NMSDataLayer(const LayerParameter& param);
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
  void InitYOLOV2Handler(const YOLOV2NMSDataParameter& param);
  void DecodeGT(const vector<Blob<Dtype>*>& bottom,
                std::vector<std::vector<int> >* gt_label,
                std::vector<std::vector<cv::Rect_<Dtype> > >* gt_bbox);
  void DecodeBBox(const Blob<Dtype>& yolo_v2_out);
  void CalcBestIOU(const std::vector<std::vector<cv::Rect_<Dtype> > >& gt_bbox);
  void CalcLocalMaxima(Blob<Dtype>& local_maxima_out);
  void CalcChannelwiseMaxIOU();

  std::vector<int> bbox_shape_;
  std::vector<std::vector<cv::Rect_<Dtype> > > bbox_;
  std::vector<std::vector<cv::Mat> > iou_;
  std::vector<cv::Mat> max_iou_;

  //Blob<cv::Rect_<Dtype> > bbox_;
  //Blob<Dtype> iou_;
  //Blob<Dtype> max_iou_;

  std::unique_ptr<bgm::AnnoDecoder<Dtype> > anno_decoder_;
  std::unique_ptr<bgm::YOLOV2Handler<Dtype> > yolo_v2_handler_;
}; // class YOLOV2NMSDataLayer

// inline functions
template <typename Dtype>
inline YOLOV2NMSDataLayer<Dtype>::YOLOV2NMSDataLayer(
    const LayerParameter& param) : Layer<Dtype>(param) {

}

template <typename Dtype>
inline const char* YOLOV2NMSDataLayer<Dtype>::type() const {
  return "YOLOV2NMSData";
}

template <typename Dtype>
inline int YOLOV2NMSDataLayer<Dtype>::ExactNumBottomBlobs() const {
  return 3; // yolo output, gt label, gt bbox
}

template <typename Dtype>
inline int YOLOV2NMSDataLayer<Dtype>::ExactNumTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline void YOLOV2NMSDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void YOLOV2NMSDataLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void YOLOV2NMSDataLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}


template <typename Dtype>
inline void YOLOV2NMSDataLayer<Dtype>::DecodeGT(
    const vector<Blob<Dtype>*>& bottom,
    std::vector<std::vector<int> >* gt_label,
    std::vector<std::vector<cv::Rect_<Dtype> > >* gt_bbox) {
  std::vector<Blob<Dtype>*> gt_blobs(bottom.begin() + 1, bottom.end());
  anno_decoder_->Decode(gt_blobs, gt_label, gt_bbox);
}
} // namespace caffe

#endif // !TLR_YOLO_V2_NMS_DATA_LAYER_HPP_
