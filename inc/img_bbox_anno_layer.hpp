#ifndef TLR_IMG_BBOX_ANNO_LAYER_HPP_
#define TLR_IMG_BBOX_ANNO_LAYER_HPP_

//#include "caffe_extend.pb.h"
#include "caffe/layers/data_layer.hpp"

namespace caffe
{

template <typename Dtype>
class ImgBBoxAnnoLayer : public caffe::DataLayer<Dtype>
{
  public:
    explicit ImgBBoxAnnoLayer(const caffe::LayerParameter& param);
    virtual void DataLayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual inline const char* type() const override;

  protected:
    ///**
    //top[0] : data (batch_size x channel x height x width)
    //top[1] : label (batch_size x 1 x 1 x 1)
    //top[2] : bbox (batch_size x 1 x max_num_bbox x 4)
    //          (bbox : (x_min, y_min, x_max, y_max)
    //*/
    //virtual void Forward_cpu(
    //    const vector<Blob<Dtype>*>& bottom,
    //    const vector<Blob<Dtype>*>& top) override;
    ///**
    //top[0] : data (batch_size x channel x height x width)
    //top[1] : label (batch_size x 1 x 1 x 1)
    //top[2] : bbox (batch_size x 1 x max_num_bbox x 4)
    //          (bbox : (x_min, y_min, x_max, y_max)
    //*/
    //virtual void Forward_gpu(
    //    const vector<Blob<Dtype>*>& bottom,
    //    const vector<Blob<Dtype>*>& top) override;
    virtual void load_batch(caffe::Batch<Dtype>* batch) override;

  private:
    void ReshpaeBatch(caffe::Batch<Dtype>* batch) const;
    void CopyImage(int item_id, 
                   const caffe::ImgBBoxAnnoDatum& datum,
                   caffe::Blob<Dtype>* batch_data) const;
    void CopyLabel(int item_id,
                   const caffe::ImgBBoxAnnoDatum& datum,
                   caffe::Blob<Dtype>* batch_label) const;
    void ComputeDataShape(vector<int>* data_shape) const;
    void ComputeLabelShape(vector<int>* label_shape) const;
    //void FowardLabelBBox_cpu(const Blob<Dtype>& batch_label,
    //                         Blob<Dtype>* label,
    //                         Blob<Dtype>* bbox) const;
    //void FowardLabelBBox_gpu(const Blob<Dtype>& batch_label,
    //                         Blob<Dtype>* label,
    //                         Blob<Dtype>* bbox) const;

    const int BATCH_SIZE_;
    const int IMG_CHANNEL_;
    const int IMG_HEIGHT_;
    const int IMG_WIDTH_;
    const int MAX_NUM_BBOX_;
}; // class ImgBBoxAnnoLayer

// inline functions
template <typename Dtype>
inline const char* ImgBBoxAnnoLayer<Dtype>::type() const {
  return "ImgBBoxAnno"; 
}

} // namespace caffe
#endif // !TLR_IMG_BBOX_ANNO_LAYER_HPP_
