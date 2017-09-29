#ifndef TLR_GT_MAP_DATA_LYAER_HPP_
#define TLR_GT_MAP_DATA_LYAER_HPP_

#include "caffe/layers/data_layer.hpp"

#include "bbox.hpp"

namespace caffe
{

template <typename Dtype>
class GTMapDataLayer : public DataLayer<Dtype>
{
  enum GTChannel {LABEL = 0, XMIN = 1, YMIN = 2, XMAX = 3, YMAX = 4};

 public:
  explicit GTMapDataLayer(const LayerParameter& param);
  virtual void DataLayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int MaxTopBlobs() const override;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void load_batch(caffe::Batch<Dtype>* batch) override;

 private:
  void ComputeMapSize(int data_width, int data_height,
                      int* width, int* height) const;
  void PrepareBatch(const std::vector<ImgBBoxAnnoDatum>& datum, Batch<Dtype>* batch);
  void CopyImage(int item_id, const ImgBBoxAnnoDatum& datum, Blob<Dtype>* batch_data);
  void CopyLabel(int item_id, const ImgBBoxAnnoDatum& datum, Blob<Dtype>* batch_label);
  void MakeLabelBBoxMap(int img_width, int img_height, const Blob<Dtype>& prefetch_label);
  void ParseLabelBBox(const Blob<Dtype>& prefetch_label,
                      std::vector<std::vector<Dtype> >* label,
                      std::vector<std::vector<bgm::BBox<Dtype> > >* bbox) const;
  int FindActiveGT(int offset_x, int offset_y, 
                   const std::vector<bgm::BBox<Dtype> >& bbox) const;
  bool IsActiveGT(const bgm::BBox<Dtype>& activation_region,
                  const bgm::BBox<Dtype>& bbox) const;
  void MakeOffsetMap(int img_width, int img_height);

  const unsigned int BATCH_SIZE_;
  const unsigned int RECEPTIVE_FIELD_WIDTH_;
  const unsigned int RECEPTIVE_FIELD_HEIGHT_;
  const unsigned int HORIZONTAL_STRIDE_;
  const unsigned int VERTICAL_STRIDE_;
  const bool PATCH_OFFSET_NORMALIZATION_;
  const bool BBOX_NORMALIZATION_; 

  const bool USE_PAD_;
  PaddingParameter::PaddingType pad_type_;
  int pad_up_;
  int pad_down_;
  int pad_left_;
  int pad_right_;

  ActivationRegionParameter::ActivationMethod activation_method_;
  bgm::BBox<Dtype> activation_region_;  

  Blob<Dtype> label_map_;
  Blob<Dtype> offset_map_;
  Blob<Dtype> bbox_map_;
}; // class GTMapDataLayer


// inline functions
template <typename Dtype>
inline const char* GTMapDataLayer<Dtype>::type() const {
  return "GTMapData";
}

template <typename Dtype>
inline int GTMapDataLayer<Dtype>::MaxTopBlobs() const {
  return 4;
}

template <typename Dtype>
inline void GTMapDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

} // namespace caffe
#endif // !TLR_GT_MAP_DATA_LYAER_HPP_
