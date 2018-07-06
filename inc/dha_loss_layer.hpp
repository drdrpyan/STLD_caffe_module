#ifndef TLR_OSO_LOSS_LAYER_HPP_
#define TLR_OSO_LOSS_LAYER_HPP_

#include "caffe/layers/loss_layer.hpp"

#include "anno_decoder.hpp"
#include "focal_loss.hpp"
//#include "box_container.hpp"

#include <opencv2/core.hpp>

#include <memory>

namespace caffe
{

template <typename Dtype>
class DHALossLayer : public LossLayer<Dtype>
{
 private:
  //struct Matching
  //{
  //  int gt_idx;
  //  int cell_idx;
  //  int anchor_idx;

  //  bool operator<(const Matching& other) {
  //    return (cell_idx == other.cell_idx) ? 
  //      (anchor_idx < other.anchor_idx) : (other_idx < other.cell_idx);
  //  }
  //};
  //struct Matching
  //{
  //  int anchor_map_idx;
  //  int gt_idx;
  //};
  //enum AnchorId {NEG = 0, POS_BEGIN = 1};
  enum AnchorElemID {X = 0, Y, W, H, CLASS_BEGIN};

 public:
  explicit DHALossLayer(const LayerParameter& param);
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
  void InitAnchorMap();
  void Clear();
  void FindBestMatch(const std::vector<cv::Rect_<Dtype> >& true_gt_box,
                     std::vector<std::pair<int, int> >* best_match) const;
  void ArgSortByCellIdx(const std::vector<std::pair<int, int> >& match,
                        std::vector<int>* sorted_idx) const;
  void ForwardPositive(const Blob<Dtype>& input, 
                       int n, int cell_r, int cell_c,
                       int true_anchor, const cv::Rect_<Dtype>& bbox,
                       int true_label);
  void ForwardNegative(const Blob<Dtype>& input,
                       int n, int cell_r, int cell_c);
  void ForwardNegAnchorScore(const Blob<Dtype>& input, int n,
                             int cell_r, int cell_c, bool target);
  void ForwardPosAnchorScore(const Blob<Dtype>& input, int n,
                             int cell_r, int cell_c, int true_anchor);

  void ForwardAnchorScore(const Blob<Dtype>& input, int n,
                          int cell_r, int cell_c, int true_anchor);
  void ForwardBBox(
    const Blob<Dtype>& input, int n, int cell_r, int cell_c, int anchor_id,
    const cv::Rect_<Dtype>& target_box_form = cv::Rect_<Dtype>(0.5, 0.5, 0., 0.));
  void ForwardClassScore(const Blob<Dtype>& input, 
                         int n, int cell_r, int cell_c, 
                         int anchor_id, int true_class,
                         bool class_idx_zero_begin = true);

  Dtype CalcIoU(const cv::Rect_<Dtype>& box1,
                const cv::Rect_<Dtype>& box2) const;
  Dtype CalcOverlap(Dtype anchor1, Dtype length1,
                    Dtype anchor2, Dtype length2) const;
  Dtype Sigmoid(Dtype value) const;

  cv::Rect_<Dtype> RawBoxToAnchorRelativeForm(
      const cv::Rect_<Dtype>& raw_box, 
      const cv::Rect_<Dtype>& anchor) const;

  int GetAnchorScoreChannel(int anchor_id) const;
  int GetAnchorElemChannel(int anchor_id, const AnchorElemID& elem) const;
  int GetClassScoreChannel(int anchor_id, int class_idx,
                           bool zero_begin = true) const;
  int GetCellIdx(int cell_r, int cell_c) const;
  void GetCellOffset(int cell_idx, int* cell_r, int* cell_c) const;

  //void InitAnchorMap() {}

  //bgm::BoxContainer<cv::Rect_<Dtype> > anchor_map_;
  Dtype noobj_scale_, obj_scale_, cls_scale_, coord_scale_;
  Dtype noobj_loss_, obj_loss_, anchor_cls_loss_, cls_loss_, coord_loss_, area_loss_;
  Dtype avg_noobj_, avg_obj_, avg_pos_anchor_, avg_neg_anchor_, avg_pos_cls_, avg_neg_cls_, avg_iou_;
  int obj_cnt_; 

  Blob<Dtype> diff_;

  int num_class_;
  cv::Size cell_size_;
  float overlap_threshold_;
  std::vector<cv::Rect_<Dtype> > anchor_;
  std::vector<float> anchor_wfl_alpha_;
  std::vector<float> anchor_wfl_gamma_;
  std::vector<float> class_wfl_alpha_;
  std::vector<float> class_wfl_gamma_;

  cv::Size map_size_;
  int num_cell_;
  int neg_anchor_id_;
  //std::vector<cv::Rect_<Dtype> > anchor_map_;
  std::vector<std::vector<cv::Rect_<Dtype> > > anchor_map_;

  bgm::FocalLoss<Dtype> focal_loss_;
  std::unique_ptr<bgm::AnnoDecoder<Dtype> > anno_decoder_;
};

// inline functions
template <typename Dtype>
inline DHALossLayer<Dtype>::DHALossLayer(
    const LayerParameter& param) : LossLayer<Dtype>(param) {

}

template <typename Dtype>
inline const char* DHALossLayer<Dtype>::type() const {
  return "DHALoss";
}

template <typename Dtype>
inline int DHALossLayer<Dtype>::ExactNumBottomBlobs() const {
  return 3;
}

template <typename Dtype>
inline int DHALossLayer<Dtype>::ExactNumTopBlobs() const {
  return -1;
}

template <typename Dtype>
inline int DHALossLayer<Dtype>::MinTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline int DHALossLayer<Dtype>::MaxTopBlobs() const {
  return 13;
}

template <typename Dtype>
inline Dtype DHALossLayer<Dtype>::Sigmoid(Dtype value) const {
  return 1. / (1. + std::exp(-value));
}

template <typename Dtype>
inline int DHALossLayer<Dtype>::GetAnchorScoreChannel(int anchor_id) const {
  CHECK_GE(anchor_id, 0);
  CHECK_LE(anchor_id, neg_anchor_id_);
  return anchor_id;
  //CHECK_LT(anchor, anchor_.size());
  //return (anchor < 0) ? 
  //  AnchorScoreCh::NEG : 
  //  static_cast<int>(AnchorScoreCh::POS_BEGIN) + anchor;
}

template <typename Dtype>
inline int DHALossLayer<Dtype>::GetAnchorElemChannel(
    int anchor_id, const AnchorElemID& elem) const {
  CHECK_GE(anchor_id, 0);
  CHECK_LT(anchor_id, anchor_.size());
  return anchor_.size() + (anchor_id * (4 + num_class_)) + static_cast<int>(elem);
}

template <typename Dtype>
inline int DHALossLayer<Dtype>::GetClassScoreChannel(
    int anchor_id, int class_idx, bool zero_begin) const {
  const int cls_idx = zero_begin ? class_idx : class_idx - 1;
  CHECK_GE(cls_idx, 0);
  CHECK_LT(cls_idx, num_class_);
  return GetAnchorElemChannel(anchor_id, AnchorElemID::CLASS_BEGIN) + cls_idx;
}

template <typename Dtype>
inline int DHALossLayer<Dtype>::GetCellIdx(int cell_r, int cell_c) const {
  CHECK_GE(cell_r, 0);
  CHECK_LT(cell_r, map_size_.height);
  CHECK_GE(cell_c, 0);
  CHECK_LT(cell_c, map_size_.width);
  return cell_r * map_size_.width + cell_c;
}

template <typename Dtype>
inline void DHALossLayer<Dtype>::GetCellOffset(
    int cell_idx, int* cell_r, int* cell_c) const {
  CHECK_GE(cell_idx, 0);
  CHECK_LT(cell_idx, anchor_map_.size());
  CHECK(cell_r);
  CHECK(cell_c);
  *cell_r = cell_idx / map_size_.width;
  *cell_c = cell_idx % map_size_.width;
}

} // namespace caffe

#endif // !TLR_OSO_LOSS_LAYER_HPP_