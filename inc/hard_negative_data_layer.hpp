#ifndef TLR_HARD_NEGATIVE_DATA_LAYER_HPP_
#define TLR_HARD_NEGATIVE_DATA_LAYER_HPP_

#include "base_img_bbox_data_layer.hpp"

namespace caffe
{

template <typename Dtype>
class HardNegativeDataLayer : public BaseImgBBoxDataLayer<Dtype>
{
 public:
  HardNegativeDataLayer(const LayerParameter& param);
  virtual const char* type() const override;
  virtual int MaxTopBlobs() const override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;

 private:

};

// inline functions
template <typename Dtype>
inline HardNegativeDataLayer<Dtype>::HardNegativeDataLayer(
    const LayerParameter& param) 
  : BaseImgBBoxDataLayer<Dtype>(param) {

}

template <typename Dtype>
inline const char* HardNegativeDataLayer<Dtype>::type() const {
  return "HardNegativeData";
}

template <typename Dtype>
inline int HardNegativeDataLayer<Dtype>::MaxTopBlobs() const {
  return 4;
}

template <typename Dtype>
inline void HardNegativeDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}
} // namespace caffe

#endif // !TLR_HARD_NEGATIVE_DATA_LAYER_HPP_
