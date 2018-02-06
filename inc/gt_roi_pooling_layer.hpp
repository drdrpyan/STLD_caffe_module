#ifndef TLR_GT_ROI_POOLING_LAYER_HPP_
#define TLR_GT_ROI_POOLING_LAYER_HPP_

#include "caffe/layer.hpp"

#include "anno_encoder.hpp"
#include "anno_decoder.hpp"
#include "uniform_integer_rng.hpp"
#include "obj_contained.hpp"

#include <opencv2/core.hpp>

#include <memory>

namespace caffe
{

template <typename Dtype>
class GTROIPoolingLayer : public Layer<Dtype>
{
  struct ROIRelation
  {
    int bot_idx;
    int top_idx;
    int offset_x;
    int offset_y;
  };

 public:
  explicit GTROIPoolingLayer(const LayerParameter& param);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int ExactNumBottomBlobs() const override;
  virtual int ExactNumTopBlobs() const override;

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;

  void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) override;

 private:
  void PickRandomROI(const std::vector<Blob<Dtype>*>& bottom,
                     std::vector<std::vector<int> >* roi_label,
                     std::vector<std::vector<cv::Rect_<Dtype> > >* roi_bbox);
  void ImgBBoxToFMBBox(const std::vector<cv::Rect_<Dtype> >& img_bbox,
                       int fm_width, int fm_height,
                       std::vector<cv::Rect_<Dtype> >* fm_bbox) const;
  cv::Point GetRandomPosROI(const std::vector<cv::Rect_<Dtype> >& fm_bbox);
  void GetContainedBBox(const cv::Rect& background,
                        const std::vector<cv::Rect_<Dtype> >& bbox,
                        std::vector<int>* idx) const;

  void MakeGTTop(const std::vector<std::vector<int> >& roi_label,
                 const std::vector<std::vector<cv::Rect_<Dtype> > >& roi_bbox,
                 Blob<Dtype>* label_top, Blob<Dtype>* bbox_top) const;
  void PoolROI_cpu(const Blob<Dtype>& bottom, Blob<Dtype>* top) const;
  void PoolROI_gpu(const Blob<Dtype>& bottom, Blob<Dtype>* top) const;

  bool pool_each_gt_;
  int num_pos_;
  int num_neg_;
  cv::Size img_size_;
  cv::Size roi_size_;
  //int stride_x_;
  //int stride_y_;

  std::vector<ROIRelation> roi_relation_;

  // util mudules
  std::unique_ptr<bgm::AnnoEncoder<Dtype> > gt_encoder_;
  std::unique_ptr<bgm::AnnoDecoder<Dtype> > gt_decoder_;
  std::shared_ptr<bgm::UniformIntegerRNG<int> > uniform_rng_;
  std::unique_ptr<bgm::ObjContained<Dtype> > obj_contained_;
  
}; // class GTROIPoolingLayer

// inline functions
template <typename Dtype>
inline GTROIPoolingLayer<Dtype>::GTROIPoolingLayer(
    const LayerParameter& param) : Layer<Dtype>(param) {

}


template <typename Dtype>
inline const char* GTROIPoolingLayer<Dtype>::type() const {
  return "GTROIPooling";
}

template <typename Dtype>
inline int GTROIPoolingLayer<Dtype>::ExactNumBottomBlobs() const {
  return 3;
}

template <typename Dtype>
inline int GTROIPoolingLayer<Dtype>::ExactNumTopBlobs() const {
  return 3;
}

} // namespace caffe

#endif // !TLR_GT_ROI_POOLING_LAYER_HPP_
