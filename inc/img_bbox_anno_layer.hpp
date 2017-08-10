#ifndef TLR_IMG_BBOX_ANNO_LAYER_HPP_
#define TLR_IMG_BBOX_ANNO_LAYER_HPP_

#include "caffe/layers/data_layer.hpp"

namespace bgm
{

template <typename Dtype>
class ImgBBoxAnnoLayer : public caffe::DataLayer<Dtype>
{
  public:
    virtual inline const char* type() const override;

  protected:
    virtual void load_batch(Batch<Dtype>* batch) override;

  private:
}; // class ImgBBoxAnnoLayer

// inline functions
template <typename Dtype>
inline const char* ImgBBoxAnnoLayer<Dtype>::type() const {
  return "ImgBBoxAnno"; 
}

} // namespace bgm
#endif // !TLR_IMG_BBOX_ANNO_LAYER_HPP_
