#ifndef TLR_ROW_COL_PROPOSAL_TRANSFORM_LAYER_HPP_
#define TLR_ROW_COL_PROPOSAL_TRANSFORM_LAYER_HPP_

#include "caffe/layer.hpp"

#include "obj_contained.hpp"

#include <opencv2/core.hpp>

namespace caffe
{

template <typename Dtype>
class RowColProposalTransformLayer : public Layer<Dtype>
{
 public:
  explicit RowColProposalTransformLayer(const LayerParameter& param);
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
  //void PickGTForOutCell(const std::vector<int>& all_label,
  //                      const std::vector<cv::Rect_<Dtype> >& all_bbox,
  //                      int out_r, int out_c,
  //                      std::vector<int>* picked_label,
  //                      std::vector<cv::Rect_<Dtype> >* picked_bbox) const;
  int GetGrid(Dtype obj_position, Dtype grid_step, Dtype origin = 0) const;
  void RCPTransform(int batch,
                    const std::vector<int>& label,
                    const std::vector<cv::Rect_<Dtype> >& bbox,
                    Blob<Dtype>* top) const;

  int row_;
  int col_;
  int in_width_;
  int in_height_;
  int out_width_;
  int out_height_;
  bool objectness_;
  int num_label_;
  std::vector<int> ignore_label_;

  bgm::ObjCenterContained<Dtype> obj_contained_;
};

// inline functions
template <typename Dtype>
inline RowColProposalTransformLayer<Dtype>::RowColProposalTransformLayer(
    const LayerParameter& param) : Layer<Dtype>(param) {

}

template <typename Dtype>
inline const char* RowColProposalTransformLayer<Dtype>::type() const {
  return "RowColProposalTransform";
}

template <typename Dtype>
inline int RowColProposalTransformLayer<Dtype>::ExactNumBottomBlobs() const {
  return 2;
}

template <typename Dtype>
inline int RowColProposalTransformLayer<Dtype>::ExactNumTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline void RowColProposalTransformLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void RowColProposalTransformLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void RowColProposalTransformLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
int RowColProposalTransformLayer<Dtype>::GetGrid(
    Dtype obj_position, Dtype grid_step, Dtype origin) const {
  CHECK_GE(obj_position, origin);
  CHECK_GT(grid_step, 0);
  return std::floor((obj_position - origin) / grid_step);
}

} // namespace caffe

#endif // !TLR_ROW_COL_PROPOSAL_TRANSFORM_LAYER_HPP_
