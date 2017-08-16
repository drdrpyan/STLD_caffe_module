#ifndef TLR_IMG_BBOX_ANNO_LAYER_HPP_
#define TLR_IMG_BBOX_ANNO_LAYER_HPP_

#include "caffe_extend.pb.h"

#include "caffe/layers/data_layer.hpp"

namespace bgm
{

template <typename Dtype>
class ImgBBoxAnnoLayer : public caffe::DataLayer<Dtype>
{
  public:
    ImgBBoxAnnoLayer(
        const caffe_ext::ExtendedLayerParameter& param);
    virtual inline const char* type() const override;

  protected:
    virtual void load_batch(caffe::Batch<Dtype>* batch) override;

  private:
    void CopyImage(int item_id, 
                   const caffe_ext::ImgBBoxAnnoDatum& datum,
                   caffe::Blob<Dtype>* batch_data) const;
    void CopyLabel(int item_id,
                   const caffe_ext::ImgBBoxAnnoDatum& datum,
                   caffe::Blob<Dtype>* batch_label) const;
}; // class ImgBBoxAnnoLayer

// inline functions
template <typename Dtype>
inline const char* ImgBBoxAnnoLayer<Dtype>::type() const {
  return "ImgBBoxAnno"; 
}

} // namespace bgm
#endif // !TLR_IMG_BBOX_ANNO_LAYER_HPP_
