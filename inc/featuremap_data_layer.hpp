#ifndef TLR_FEATUREMAP_DATA_LAYER_HPP_
#define TLR_FEATUREMAP_DATA_LAYER_HPP_

#include "caffe/layers/data_layer.hpp"

namespace caffe
{

template <typename Dtype>
class FeaturemapDataLayer : public DataLayer<Dtype>
{
 public:
  explicit FeaturemapDataLayer(const LayerParameter& param);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;

 protected:
  virtual void load_batch(Batch<Dtype>* batch) override;

};

// inline functions
template <typename Dtype>
FeaturemapDataLayer<Dtype>::FeaturemapDataLayer(const LayerParameter& param) 
  : DataLayer<Dtype>(param) {

}

template <typename Dtype>
const char* FeaturemapDataLayer<Dtype>::type() const {
  return "FeaturemapData";
}

} // namespace caffe

#endif // !TLR_FEATUREMAP_DATA_LAYER_HPP_
