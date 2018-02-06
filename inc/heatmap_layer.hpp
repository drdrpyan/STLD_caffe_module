#ifndef TLR_HEATMAP_LAYER_HPP_
#define TLR_HEATMAP_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class HeatmapLayer : public Layer<Dtype>
{

 public:
  HeatmapLayer(const LayerParameter& param);
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
  void GetCellPos(Dtype x, Dtype y, int* r, int* c) const;
  void GetNewBBox(Dtype src_x, Dtype src_y, 
                  Dtype src_w, Dtype src_h,
                  int pos_r, int poc_c,
                  Dtype* dst_x, Dtype* dst_y,
                  Dtype* dst_w, Dtype* dst_h) const;
  //void GetCenter(Dtype x, Dtype y, Dtype w, Dtype h,
  //               Dtype* center_x, Dtype* center_y) const;

  int num_label_;
  bool bbox_normalized_;
  Dtype width_;
  Dtype height_;
  int rows_;
  int cols_;

  Dtype x_step_;
  Dtype y_step_;
}; // class HeatmapLayer

// inline functions
template <typename Dtype>
inline HeatmapLayer<Dtype>::HeatmapLayer(const LayerParameter& param) 
  : Layer<Dtype>(param) {

}

template <typename Dtype>
inline const char* HeatmapLayer<Dtype>::type() const {
  return "Heatmap";
}

template <typename Dtype>
inline int HeatmapLayer<Dtype>::ExactNumBottomBlobs() const {
  return 2;
}

template <typename Dtype>
inline int HeatmapLayer<Dtype>::ExactNumTopBlobs() const {
  return 2;
}

template <typename Dtype>
inline void HeatmapLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void HeatmapLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void HeatmapLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}
} // namespace caffe
#endif // !TLR_HEATMAP_LAYER_HPP_
