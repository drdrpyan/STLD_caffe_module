#ifndef TLR_GRID_PROPOSAL_LOSS_LAYER_HPP_
#define TLR_GRID_PROPOSAL_LOSS_LAYER_HPP_

#include "caffe/layers/loss_layer.hpp"

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"

namespace caffe
{

template <typename Dtype>
class GridProposalLossLayer : public LossLayer<Dtype>
{
 public:
  explicit GridProposalLossLayer(const LayerParameter& param);
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
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom) override;

 private:
  void TransformGT(const Blob<Dtype>& gt,
                   Blob<Dtype>* transformed_gt) const;
  void TransformGTRow(
      const Dtype* gt_data, int h, int w,
      Dtype* transformed) const;
  void TransformGTRow(
      const Dtype* gt_data, int h, int w, Dtype label,
      Dtype* transformed) const;
  void TransformGTCol(
      const Dtype* gt_data, int h, int w,
      Dtype* transformed) const;
  void TransformGTCol(
      const Dtype* gt_data, int h, int w, Dtype label,
      Dtype* transformed) const;
  void ComputeDiffLoss(const Blob<Dtype>& input,
                        const Blob<Dtype>& transf_gt,
                        Dtype label = 1);
  
  float noobj_weight_;
  std::vector<float> obj_weight_;
  int num_class_;
  bool general_obj_proposal_;

  
  std::vector<Dtype> obj_loss_;
  std::vector<int> obj_count_;
  Dtype noobj_loss_;
  int noobj_count_;

  Dtype avg_obj_conf_, avg_noobj_conf_;

  Blob<Dtype> diff_;

  shared_ptr<SigmoidCrossEntropyLossLayer<Dtype> > loss_layer_;
  //shared_ptr<Blob<Dtype> > loss_output_;
  vector<Blob<Dtype>*> loss_bottom_vec_;
  vector<Blob<Dtype>*> loss_top_vec_;
  Blob<Dtype> transf_gt_;
};

// inline functions
template <typename Dtype>
inline const char* GridProposalLossLayer<Dtype>::type() const {
  return "GridProposalLoss";
}

template <typename Dtype>
inline int GridProposalLossLayer<Dtype>::ExactNumBottomBlobs() const {
  return 2;
}

template <typename Dtype>
inline int GridProposalLossLayer<Dtype>::ExactNumTopBlobs() const {
  return -1;
}

} // namespace caffe
#endif // !TLR_GRID_PROPOSAL_LOSS_LAYER_HPP_

