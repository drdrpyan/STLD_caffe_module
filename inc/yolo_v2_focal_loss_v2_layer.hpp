#ifndef TLR_YOLO_V2_FOCAL_LOSS_V2_LAYER_HPP_
#define TLR_YOLO_V2_FOCAL_LOSS_V2_LAYER_HPP_

#include "caffe/layers/loss_layer.hpp"

#include "anno_decoder.hpp"

#include <opencv2/core.hpp>

#include <memory>

namespace caffe
{

template <typename Dtype>
class YOLOV2FocalLossV2Layer : public LossLayer<Dtype>
{
  enum {NUM_ANCHOR_ELEM = 5};
  enum AnchorChannel {X = 0, Y, W, H, CONF, CLASS_BEGIN};

 public:
  explicit YOLOV2FocalLossV2Layer(const LayerParameter& param);
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
  void Clear();
  void ShiftAnchors(int cell_row, int cell_col, 
                    std::vector<cv::Rect_<Dtype> >* shifted) const;
  void FindBestMatching(const std::vector<cv::Rect_<Dtype> >& anchor,
                        const std::vector<cv::Rect_<Dtype> >& gt_bbox,
                        int* best_anchor_idx, int* best_bbox_idx, 
                        Dtype* best_iou) const;
  void FindBestMatching(const cv::Rect_<Dtype>& anchor,
                        const std::vector<cv::Rect_<Dtype> >& gt_bbox,
                        int* best_bbox_idx, Dtype* best_iou) const;
  cv::Rect_<Dtype> RawBBoxToAnchorBBox(const cv::Rect_<Dtype>& raw_bbox,
                                       const cv::Rect_<Dtype>& anchor) const;
  void ForwardNegative(const Blob<Dtype>& input, 
                       int n, int h, int w, int anchor);
  void ForwardPositive(const Blob<Dtype>& input, 
                       int n, int h, int w, int anchor,
                       Dtype true_label, const cv::Rect_<Dtype>& true_bbox,
                       Dtype iou);
  void ForwardConf(const Blob<Dtype>& input,
                   int n, int h, int w, int anchor, Dtype scale, Dtype iou,
                   Dtype& loss, Dtype& sum_conf);
  void ForwardBBox(
      const Blob<Dtype>& input, int n, int h, int w, int anchor,
      Dtype scale,
      const cv::Rect_<Dtype>& target_yolo_form = cv::Rect_<Dtype>(0.5, 0.5, 0., 0.));
  void ForwardClass(const Blob<Dtype>& input,
                    int n, int h, int w, int anchor, int true_label);
  cv::Point GetCellTopLeft(int row, int col) const;
  Dtype CalcIoU(const cv::Rect_<Dtype>& box1,
                const cv::Rect_<Dtype>& box2) const;
  Dtype CalcOverlap(Dtype anchor1, Dtype length1,
                    Dtype anchor2, Dtype length2) const;
  

  int GetAnchorChannel(int anchor, AnchorChannel ch) const;
  int GetClassChannel(int anchor, int class_label) const;

  cv::Rect_<Dtype> GetPredBBox(const Blob<Dtype>& input,
                               int n, int h, int w, int anchor) const;
  cv::Rect_<Dtype> RawBoxToYOLOBox(const cv::Rect_<Dtype>& raw_box,
                                   const cv::Rect_<Dtype>& anchor) const;
  cv::Rect_<Dtype> YOLOBoxToRawBox(const cv::Rect_<Dtype>& yolo_box,
                                   const cv::Rect_<Dtype>& anchor,
                                   bool shift = true) const;
  Dtype Sigmoid(Dtype value) const;

  int num_anchor_;
  int num_class_;
  
  cv::Size img_size_;
  cv::Size yolo_map_size_;
  float overlap_threshold_;

  Dtype noobj_scale_, obj_scale_, cls_scale_, coord_scale_;

  Dtype noobj_loss_, obj_loss_, cls_loss_, coord_loss_, area_loss_;
  Dtype avg_noobj_, avg_obj_, avg_pos_cls_, avg_neg_cls_, avg_iou_;
  int obj_cnt_;

  Dtype max_noobj_, min_obj_;

  std::vector<cv::Rect_<Dtype> > anchor_;
  Blob<Dtype> diff_;

  std::unique_ptr<bgm::AnnoDecoder<Dtype> > anno_decoder_;

  float focal_loss_alpha_;
  float focal_loss_gamma_;
}; // class YOLOV2LossLayer

// inline functions
template <typename Dtype>
inline YOLOV2FocalLossV2Layer<Dtype>::YOLOV2FocalLossV2Layer(
    const LayerParameter& param) : LossLayer<Dtype>(param) {

}

template <typename Dtype>
inline const char* YOLOV2FocalLossV2Layer<Dtype>::type() const {
  return "YOLOV2FocalLossV2";
}

template <typename Dtype>
inline int YOLOV2FocalLossV2Layer<Dtype>::ExactNumBottomBlobs() const {
  return 3;
}

template <typename Dtype>
inline int YOLOV2FocalLossV2Layer<Dtype>::ExactNumTopBlobs() const {
  return -1;
}

template <typename Dtype>
inline int YOLOV2FocalLossV2Layer<Dtype>::MinTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline int YOLOV2FocalLossV2Layer<Dtype>::MaxTopBlobs() const {
  return 13;
}


template <typename Dtype>
inline int YOLOV2FocalLossV2Layer<Dtype>::GetAnchorChannel(
    int anchor, AnchorChannel ch) const {
  CHECK_GE(anchor, 0);
  CHECK_LT(anchor, anchor_.size());
  return anchor * (NUM_ANCHOR_ELEM + num_class_) + static_cast<int>(ch);
}

template <typename Dtype>
inline int YOLOV2FocalLossV2Layer<Dtype>::GetClassChannel(
    int anchor, int class_label) const {
  CHECK_GE(anchor, 0);
  CHECK_LT(anchor, anchor_.size());
  CHECK_GE(class_label, 1);
  CHECK_LE(class_label, num_class_);
  return anchor * (NUM_ANCHOR_ELEM + num_class_) + NUM_ANCHOR_ELEM + (class_label - 1);
}

template <typename Dtype>
inline Dtype YOLOV2FocalLossV2Layer<Dtype>::Sigmoid(Dtype value) const {
  return 1. / (1. + std::exp(-value));
}

} // namespace caffe

#endif // !TLR_YOLO_V2_FOCAL_LOSS_V2_LAYER_HPP_