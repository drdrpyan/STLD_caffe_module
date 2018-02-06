#ifndef TLR_BASE_IMG_BBOX_DATA_LAYER_HPP_
#define TLR_BASE_IMG_BBOX_DATA_LAYER_HPP_

#include "caffe/layers/data_layer.hpp"

#include "bbox.hpp"
#include "img_augmentation.hpp"

namespace caffe
{

template <typename Dtype>
class BaseImgBBoxDataLayer : public DataLayer<Dtype>
{
 public:
  enum GTChannel { LABEL = 0, XMIN = 1, YMIN = 2, XMAX = 3, YMAX = 4 };

 public:
  explicit BaseImgBBoxDataLayer(const LayerParameter& param);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int MinTopBlobs() const override;
  virtual int MaxTopBlobs() const override;

  void ParseLabelBBox(const Blob<Dtype>& prefetch_label,
                      std::vector<std::vector<Dtype> >* label,
                      std::vector<std::vector<bgm::BBox<Dtype> > >* bbox) const;
#ifdef USE_OPENCV
  void ParseLabelBBox(const Blob<Dtype>& prefetch_label,
                      std::vector<std::vector<Dtype> >* label,
                      std::vector<std::vector<cv::Rect_<Dtype> > >* bbox) const;
#endif // USE_OPENCV

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void load_batch(caffe::Batch<Dtype>* batch) override;


  // deprecated
  void SetPad(const PaddingParameter& pad_param);
  void NoPad();

 private:
  void PrepareBatch(const std::vector<ImgBBoxAnnoDatum>& datum,
                    Batch<Dtype>* batch);
  void CopyImage(int item_id, const ImgBBoxAnnoDatum& datum,
                 Blob<Dtype>* batch_data);
  void CopyLabel(int item_id, const ImgBBoxAnnoDatum& datum,
                 Blob<Dtype>* batch_label);

  void PrepareBatch(const std::vector<cv::Mat>& img,
                    const std::vector<std::vector<int> >& label,
                    Batch<Dtype>* batch);
  void PrepareBatch(const std::vector<cv::Mat>& img,
                    Batch<Dtype>* batch);
  void CopyImage(int item_id, const cv::Mat& img,
                 Blob<Dtype>* batch_data);
  void CopyLabel(int item_id, const std::vector<int>& label,
                 const std::vector<cv::Rect2f>& bbox,
                 Blob<Dtype>* batch_label);

  // deprecated, use img_aug_ instead
  bool use_pad_;
  PaddingParameter::PaddingType pad_type_;
  int pad_up_;
  int pad_down_;
  int pad_left_;
  int pad_right_;

  bool do_augment_;
  ImgAugmentation img_aug_;

}; // class BaseImgBBoxDataLayer

// inline functions
template <typename Dtype>
inline void BaseImgBBoxDataLayer<Dtype>::NoPad() {
  use_pad_ = false;
}

template <typename Dtype>
inline const char* BaseImgBBoxDataLayer<Dtype>::type() const {
  return "BaseImgBBoxData";
}

template <typename Dtype>
inline int BaseImgBBoxDataLayer<Dtype>::MinTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline int BaseImgBBoxDataLayer<Dtype>::MaxTopBlobs() const {
  return 2;
}

} // namespace caffe

#endif // !TLR_BASE_IMG_BBOX_DATA_LAYER_HPP_
