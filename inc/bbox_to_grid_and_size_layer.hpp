#ifndef TLR_BBOX_TO_GRID_AND_SIZE_LAYER_HPP_
#define TLR_BBOX_TO_GRID_AND_SIZE_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class BBoxToGridAndSizeLayer : public Layer<Dtype>
{
 public:
  explicit BBoxToGridAndSizeLayer(const LayerParameter& param);
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

  /**
  This layer handles input data. It does not backprogate.
  */
  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) override;
  /**
  This layer handles input data. It does not backprogate.
  */
  virtual void Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) override;

 private:
  int GetGridIdx(Dtype x_min, Dtype y_min, Dtype width, Dtype height) const;
  void GetBBoxCenter(Dtype x_min, Dtype y_min, Dtype width, Dtype height,
                     Dtype* center_x, Dtype* center_y) const;
  Dtype GetSize(Dtype size) const;

  BBoxParameter::BBoxType bbox_type_;
  std::vector<Dtype> x_grid_;
  std::vector<Dtype> y_grid_;
  std::vector<Dtype> size_grid_;
};

// inline functions
template <typename Dtype>
inline const char* BBoxToGridAndSizeLayer<Dtype>::type() const {
  return "BBoxToGridAndSize";
}

template <typename Dtype>
inline int BBoxToGridAndSizeLayer<Dtype>::ExactNumBottomBlobs() const {
  return 1;
}

template <typename Dtype>
inline int BBoxToGridAndSizeLayer<Dtype>::ExactNumTopBlobs() const {
  return 2;
}

template <typename Dtype>
inline void BBoxToGridAndSizeLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void BBoxToGridAndSizeLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void BBoxToGridAndSizeLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void BBoxToGridAndSizeLayer<Dtype>::GetBBoxCenter(
    Dtype x_min, Dtype y_min, Dtype width, Dtype height,
    Dtype* center_x, Dtype* center_y) const {
  CHECK(center_x);
  CHECK(center_y);

  *center_x = x_min + (width / 2.0);
  *center_y = y_min + (height / 2.0);
}

} // namespace caffe
#endif // !TLR_BBOX_TO_GRID_AND_SIZE_LAYER_HPP_
