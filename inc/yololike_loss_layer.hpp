#ifndef TLR_YOLOLIKE_LOSS_LAYER_HPP_
#define TLR_YOLOLIKE_LOSS_LAYER_HPP_

#include "caffe/layers/loss_layer.hpp"

#include <vector>

namespace caffe
{

template <typename Dtype>
class YOLOLikeLossLayer : public LossLayer<Dtype>
{
  enum {BBOX_CHANNELS = 5};
  enum BBoxAttr {X = 0, Y, W, H, CONF};

 public:
  explicit YOLOLikeLossLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;

  virtual const char* type() const override;

  virtual int ExactNumBottomBlobs() const override;
  virtual int MinBottomBlobs() const override;
  virtual int MaxBottomBlobs() const override;
  virtual int ExactNumTopBlobs() const override;
  virtual int MinTopBlobs() const override;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom) override;
 private:
  void GetGT(const vector<Blob<Dtype>*>& bottom,
             const Blob<Dtype>** true_label,
             const Blob<Dtype>** true_bbox) const;
  void Clear();
  void ForwardNegative(const Blob<Dtype>& input, int n, int h, int w);
  void ForwardPositive(const Blob<Dtype>& input, int n, int h, int w,
                       Dtype true_label, 
                       const std::vector<Dtype>& true_bbox);

  void ComputeNoobjLossDiff(int bbox_conf_offset,
                            const Dtype* input_data, Dtype* diff_data);
  void ComputeObjLossDiff(int bbox_conf_offset, Dtype best_iou,
                          const Dtype* input_data, Dtype* diff_data);
  void ComputeClassConfLossDiff(int class_conf_offset,
                                Dtype prediction, Dtype true_label,
                                const Dtype* input_data, Dtype* diff_data);
  void ComputeCoordAreaLossDiff(
      const std::vector<Dtype>& true_bbox,
      const std::vector<int>& best_bbox_offsets,
      const Dtype* input_data, Dtype* diff_data);

  void FindBestBBox(const Blob<Dtype>& input, int n, int h, int w,
                    const std::vector<Dtype>& true_bbox,
                    int* best_bbox_idx, Dtype* best_iou,
                    std::vector<int>* best_bbox_offsets);

  Dtype CalcIoU(const std::vector<Dtype>& box1,
                const std::vector<Dtype>& box2) const;
  Dtype CalcOverlap(Dtype anchor1, Dtype length1,
                    Dtype anchor2, Dtype length2) const;
  Dtype CalcRMSE(const std::vector<Dtype>& box1,
                 const std::vector<Dtype>& box2) const;

  int BBoxChannel(int bbox_idx, 
                  BBoxAttr bbox_attr = BBoxAttr::X) const;
  int BBoxOffset(const Blob<Dtype>& input, int n, int bbox_idx,
                 BBoxAttr bbox_attr = BBoxAttr::X, 
                 int h = 0, int w = 0) const;
  int BBoxOffset(const Blob<Dtype>& input, int bbox_idx,
                 BBoxAttr bbox_attr = BBoxAttr::X, 
                 int h = 0, int w = 0) const;

  int ClassChannel(int c = 1) const;
  int ClassConfOffset(const Blob<Dtype>& input, int n, int label,
                      int h = 0, int w = 0) const;

  float ClassWeight(int label) const;

  const int NUM_BBOX_PER_CELL_;
  const int NUM_CLASS_;

  const float NOOBJ_SCALE_;
  const float OBJ_SCALE_;
  const float CLASS_SCALE_;
  const float COORD_SCALE_;

  BBoxParameter::BBoxAnchor bbox_anchor_;

  std::vector<float> class_weight_;
  
  int grid_width_;
  int grid_height_;
  int num_cells_;

  Blob<Dtype> diff_;

  int obj_count_;
  int noobj_count_;
  Dtype noobj_loss_, obj_loss_, class_loss_, coord_loss_, area_loss_;
  Dtype noobj_conf_, obj_conf_, neg_class_conf_, pos_class_conf_, iou_;

  std::vector<Dtype> noobj_conf_list_, obj_conf_list_;
}; // class YOLOLikeLossLayer


// inline functions
template <typename Dtype>
inline const char* YOLOLikeLossLayer<Dtype>::type() const {
  return "YOLOLikeLoss";
}

template <typename Dtype>
inline int YOLOLikeLossLayer<Dtype>::ExactNumBottomBlobs() const {
  return -1;
}

template <typename Dtype>
inline int YOLOLikeLossLayer<Dtype>::MinBottomBlobs() const {
  return 2;
}

template <typename Dtype>
inline int YOLOLikeLossLayer<Dtype>::MaxBottomBlobs() const {
  return 3;
}

template <typename Dtype>
inline int YOLOLikeLossLayer<Dtype>::ExactNumTopBlobs() const {
  return -1;
}

template <typename Dtype>
inline int YOLOLikeLossLayer<Dtype>::MinTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline Dtype YOLOLikeLossLayer<Dtype>::CalcRMSE(
    const std::vector<Dtype>& box1,
    const std::vector<Dtype>& box2) const {
  CHECK_EQ(box1.size(), 4);
  CHECK_EQ(box2.size(), 4);
  return std::sqrt(std::pow(box1[0] - box2[0], 2)
                     + std::pow(box1[1] - box2[1], 2)
                     + std::pow(box1[2] - box2[2], 2)
                     + std::pow(box1[3] - box2[3], 2));
}

template <typename Dtype>
inline int YOLOLikeLossLayer<Dtype>::BBoxChannel(
    int bbox_idx, BBoxAttr bbox_attr) const {
  CHECK_GE(bbox_idx, 0);
  CHECK_LT(bbox_idx, NUM_BBOX_PER_CELL_);

  return BBOX_CHANNELS * bbox_idx + static_cast<int>(bbox_attr);
}

template <typename Dtype>
inline int YOLOLikeLossLayer<Dtype>::BBoxOffset(
    const Blob<Dtype>& input, int n, int bbox_idx,
    BBoxAttr bbox_attr, int h, int w) const {
  int ch = BBoxChannel(bbox_idx, bbox_attr);
  return input.offset(n, ch, h, w);
}

template <typename Dtype>
inline int YOLOLikeLossLayer<Dtype>::BBoxOffset(
    const Blob<Dtype>& input, int bbox_idx,
    BBoxAttr bbox_attr, int h, int w) const {
  return BBoxOffset(input, 0, bbox_idx, bbox_attr, h, w);
}

template <typename Dtype>
int YOLOLikeLossLayer<Dtype>::ClassChannel(int c) const {
  CHECK_GT(c, 0) << "label start at 1";
  CHECK_LE(c, NUM_CLASS_);

  int class_ch_begin = BBOX_CHANNELS * NUM_BBOX_PER_CELL_;
  return class_ch_begin + c - 1;
}

template <typename Dtype>
inline int YOLOLikeLossLayer<Dtype>::ClassConfOffset(
    const Blob<Dtype>& input, int n, int label, int h, int w) const {
  return input.offset(n, ClassChannel(label), h, w);
}

template <typename Dtype>
inline float YOLOLikeLossLayer<Dtype>::ClassWeight(int label) const {
  return class_weight_[label - 1];
}

} // namespace caffe

#endif // !TLR_YOLOLIKE_LOSS_LAYER_HPP_