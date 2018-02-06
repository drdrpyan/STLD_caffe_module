#ifndef TLR_GT_SUBMAP_DATA_LAYER_HPP_
#define TLR_GT_SUBMAP_DATA_LAYER_HPP_

#include "base_img_bbox_data_layer.hpp"

#include <list>
#include <deque>
#include <memory>
#include <random>

namespace caffe
{

template <typename Dtype>
class GTSubmapDataLayer : public BaseImgBBoxDataLayer<Dtype>
{
  struct TopBlob
  {
    Blob<Dtype> data;
    Blob<Dtype> label;
    Blob<Dtype> bbox;
    Blob<Dtype> offset;
  };

 public:
  explicit GTSubmapDataLayer(const LayerParameter& param);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int MinTopBlobs() const override;
  virtual int MaxTopBlobs() const override;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
 private:
  void InitBaseOffsetMap();
  void ExtractSubmap();
  void PickPositive(int num,
                    const std::vector<bgm::BBox<Dtype> >& gt_bbox,
                    int img_width, int img_height,
                    std::vector<bgm::BBox<int> >* roi) const;
  void GetCenterActivationPatchRange(
      const bgm::BBox<Dtype>& gt, int img_width, int img_height,
      int* x_min, int* x_max, int* y_min, int* y_max) const;
  void PickSemiPositive(int num,
                        const std::vector<bgm::BBox<Dtype> >& gt_bbox,
                        int img_width, int img_height,
                        std::vector<bgm::BBox<int> >* roi) const;
  void GetSemiPositiveRange(const bgm::BBox<Dtype>& gt,
                            int img_width, int img_height,
                            int* x_min, int* x_max, 
                            int* y_min, int* y_max) const;
  void PickNegative(int num_neg, int img_width, int img_height,
                    std::vector<bgm::BBox<int> >* roi) const;
  void GetUniformRandom(int num, int min, int max, 
                        std::vector<int>* random) const;
  void MakeTopBlob(int data_id,
                   const std::vector<Dtype>& gt_label,
                   const std::vector<bgm::BBox<Dtype> >& gt_bbox,
                   const Blob<Dtype>& data, const bgm::BBox<int>& roi,
                   TopBlob* top_blob) const;
  void MakeTopData(int data_id, 
                   const Blob<Dtype>& data, const bgm::BBox<int>& roi,
                   Blob<Dtype>* top_data) const;
  void MakeLabelBBoxMap(const std::vector<Dtype>& gt_label,
                        const std::vector<bgm::BBox<Dtype> >& gt_bbox,
                        const bgm::BBox<int>& roi,
                        Blob<Dtype>* label_map,
                        Blob<Dtype>* bbox_map) const;
  int FindActiveGT(int offset_x, int offset_y,
                    const std::vector<bgm::BBox<Dtype> >& bbox) const;
  bool IsActiveGT(const bgm::BBox<Dtype>& activation_region,
                  const bgm::BBox<Dtype>& bbox) const;
  void MakeOffsetMap(int img_width, int img_height,
                     const bgm::BBox<int>& roi,
                     Blob<Dtype>* offset_map) const;
  void CopyTop(int batch_idx, const TopBlob& top_blob,
               const std::vector<Blob<Dtype>*>& top) const;

  const unsigned int SUBMAP_BATCH_SIZE_;
  const unsigned int SUBMAP_WIDTH_;
  const unsigned int SUBMAP_HEIGHT_;
  const unsigned int RECEPTIVE_FIELD_WIDTH_;
  const unsigned int RECEPTIVE_FIELD_HEIGHT_;
  const unsigned int HORIZONTAL_STRIDE_;
  const unsigned int VERTICAL_STRIDE_;

  const unsigned int NUM_JITTER_;

  const bool BBOX_NORMALIZATION_;

  OffsetParameter::Anchor OFFSET_ORIGIN_;
  OffsetParameter::Anchor OFFSET_ANCHOR_;
  const bool OFFSET_NORMALIZATION_;

  ActivationRegionParameter::ActivationMethod activation_method_;
  bgm::BBox<Dtype> activation_region_;  

  //std::vector<std::shared_ptr<TopBlob> > top_queue_;
  //std::list<std::shared_ptr<TopBlob> > top_queue_;
  std::deque<std::shared_ptr<TopBlob> > top_queue_;

  std::mt19937 random_engine_;
  Blob<Dtype> base_offset_map_;
};

// inline function
template <typename Dtype>
inline const char* GTSubmapDataLayer<Dtype>::type() const {
  return "GTSubmapData";
}

template <typename Dtype>
inline int GTSubmapDataLayer<Dtype>::MinTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline int GTSubmapDataLayer<Dtype>::MaxTopBlobs() const {
  return 4;
}

template <typename Dtype>
inline void GTSubmapDataLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}
} // namespace caffe
#endif // !TLR_GT_SUBMAP_DATA_LAYER_HPP_