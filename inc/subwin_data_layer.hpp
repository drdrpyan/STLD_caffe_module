#ifndef TLR_SUBWIN_DATA_LAYER_HPP_
#define TLR_SUBWIN_DATA_LAYER_HPP_

#include "base_img_bbox_data_layer.hpp"

#include "anno_encoder.hpp"
#include "obj_contained.hpp"

namespace caffe
{

template <typename Dtype>
class SubwinDataLayer : public BaseImgBBoxDataLayer<Dtype>
{
 public:
  explicit SubwinDataLayer(const LayerParameter& param);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) override;
  //virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
  //                     const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int MinTopBlobs() const override;
  virtual int MaxTopBlobs() const override;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype> *> &bottom) override;

 private:
  void ReshapeTop(const Blob<Dtype>& src_data,
                  const Blob<Dtype>& src_label,
                  const std::vector<Blob<Dtype>*>& top) const;
  void ForwardCroppedImg_cpu(const Blob<Dtype>& src, Blob<Dtype>* dst) const;
  void ForwardCroppedImg_gpu(const Blob<Dtype>& src, Blob<Dtype>* dst) const;
  void ForwardLabelBBox(const Blob<Dtype>& src, 
                        Blob<Dtype>* label_dst, Blob<Dtype>* bbox_dst) const;
  void ForwardWholeImg(const Blob<Dtype>& src, Blob<Dtype>* dst) const;

  cv::Size win_size_;
  std::vector<cv::Point> win_offset_;
  bool global_detection_;  

  std::unique_ptr<bgm::AnnoEncoder<Dtype> > anno_encoder_;
  std::unique_ptr<bgm::ObjContained<Dtype> > obj_contained_;
};

// inline functions
template <typename Dtype>
SubwinDataLayer<Dtype>::SubwinDataLayer(const LayerParameter& param)
  : BaseImgBBoxDataLayer<Dtype>(param) {

}

template <typename Dtype>
inline const char* SubwinDataLayer<Dtype>::type() const {
  return "SubwinData";
}

template <typename Dtype>
inline int SubwinDataLayer<Dtype>::MinTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline int SubwinDataLayer<Dtype>::MaxTopBlobs() const {
  return 4;
}

template <typename Dtype>
inline void SubwinDataLayer<Dtype>::ForwardWholeImg(
    const Blob<Dtype>& src, Blob<Dtype>* dst) const {
  dst->ShareData(src);
}

} // namespace caffe

#endif // !TLR_SUBWIN_DATA_LAYER_HPP_
