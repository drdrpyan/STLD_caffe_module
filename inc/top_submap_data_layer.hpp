#ifndef TLR_TOP_SUBMAP_DATA_LAYER_HPP_
#define TLR_TOP_SUBMAP_DATA_LAYER_HPP_

#include "base_img_bbox_data_layer.hpp"

#include <list>

namespace caffe
{

template <typename Dtype>
class TopSubmapDataLayer : public BaseImgBBoxDataLayer<Dtype>
{
  struct TopBlob
  {
    Blob<Dtype> data;
    Blob<Dtype> label;
    Blob<Dtype> bbox;
    Blob<Dtype> offset;
  };

 public:
  explicit TopSubmapDataLayer(const LayerParameter& param);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  void InitUnitParam(const TopSubmapDataParameter& param);
  void InitNumSubmaps(const TopSubmapDataParameter& param);
  void InitSubmapSize();
  void InitOffsetMaps();
  void InitBaseOffsetMap(Blob<Dtype>* base);
  void MallocOffsetMap();
  void InitBBoxParam(const TopSubmapDataParameter& param);

  Blob<Dtype>& GetOffsetMap(int row, int col);
  const Blob<Dtype>& GetOffsetMap(int row, int col) const;

  void ExtractSubmap();
  void CropImg(const Blob<Dtype>& src, int idx,
               int roi_x, int roi_y, int roi_width, int roi_height,
               Blob<Dtype>* dst) const;
  bool MakeLabelBBoxMap(const std::vector<Dtype>& gt_label,
                        const std::vector<bgm::BBox<Dtype> >& gt_bbox,
                        int roi_x, int roi_y, int roi_width, int roi_height,
                        Blob<Dtype>* label_map, Blob<Dtype>* bbox_map) const;
  int FindActiveGT(int offset_x, int offset_y,
                   const std::vector<bgm::BBox<Dtype> >& bbox) const;

  void CopyTop(int batch_idx, const TopBlob& top_blob,
               const std::vector<Blob<Dtype>*>& top) const;

  const unsigned int IMG_WIDTH_;
  const unsigned int IMG_HEIGHT_;
  const unsigned int SUBMAP_BATCH_SIZE_;
  const unsigned int WINDOW_WIDTH_;
  const unsigned int WINDOW_HEIGHT_;
  const unsigned int WIN_HORIZONTAL_STRIDE_;
  const unsigned int WIN_VERTICAL_STRIDE_;

  const bool OFFSET_NORMALIZE_;

  unsigned int offset_unit_width_;
  unsigned int offset_unit_height_;
  unsigned int offset_h_stride_;
  unsigned int offset_v_stride_;

  unsigned int gt_unit_width_;
  unsigned int gt_unit_height_;
  unsigned int gt_h_stride_;
  unsigned int gt_v_stride_;

  //unsigned int cell_width_;
  //unsigned int cell_height_;
  //unsigned int cell_horizontal_stride_;
  //unsigned int cell_vertical_stride_;

  int num_submap_rows_, num_submap_cols_;
  int offset_submap_width_, offset_submap_height_;
  int gt_submap_width_, gt_submap_height_;

  BBoxParameter::BBoxAnchor bbox_anchor_;
  bool bbox_normalize_;

  std::vector<std::unique_ptr<Blob<Dtype> > > offset_map_;

  std::list<std::shared_ptr<TopBlob> > top_queue_;
};

// inline function
template <typename Dtype>
inline const char* TopSubmapDataLayer<Dtype>::type() const {
  return "TopSubmapData";
}

template <typename Dtype>
inline int TopSubmapDataLayer<Dtype>::MinTopBlobs() const {
  return 2;
}

template <typename Dtype>
inline int TopSubmapDataLayer<Dtype>::MaxTopBlobs() const {
  return 4;
}

template <typename Dtype>
inline void TopSubmapDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void TopSubmapDataLayer<Dtype>::InitSubmapSize() {
  offset_submap_width_ = ((WINDOW_WIDTH_ - offset_unit_width_) / offset_h_stride_) + 1;
  offset_submap_height_ = ((WINDOW_WIDTH_ - offset_unit_height_) / offset_v_stride_) + 1;

  gt_submap_width_ = ((WINDOW_WIDTH_ - gt_unit_width_) / gt_h_stride_) + 1;
  gt_submap_height_ = ((WINDOW_WIDTH_ - gt_unit_height_) / gt_v_stride_) + 1;

  //submap_width_ = ((WINDOW_WIDTH_ - cell_width_) / cell_horizontal_stride_) + 1;
  //submap_height_ = ((WINDOW_HEIGHT_ - cell_height_) / cell_vertical_stride_) + 1;
}

template <typename Dtype>
inline void TopSubmapDataLayer<Dtype>::MallocOffsetMap() {
  offset_map_.clear();
  for (int i = 0; i < num_submap_rows_*num_submap_cols_; ++i)
    offset_map_.push_back(std::unique_ptr<Blob<Dtype> >(new Blob<Dtype>));
}

template <typename Dtype>
inline Blob<Dtype>& TopSubmapDataLayer<Dtype>::GetOffsetMap(int row, int col) {
  const TopSubmapDataLayer<Dtype>* const_this = this;
  return const_cast<Blob<Dtype>&>(const_this->GetOffsetMap(row, col));
}



} // namespace caffe

#endif // !TLR_TOP_SUBMAP_DATA_LAYER_HPP_
