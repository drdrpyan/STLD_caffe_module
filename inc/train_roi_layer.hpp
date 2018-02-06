#ifndef TLR_TRAIN_ROI_LAYER_HPP_
#define TLR_TRAIN_ROI_LAYER_HPP_

#include "anno_decoder.hpp"
#include "train_roi_generator.hpp"
#include "roi_extractor.hpp"

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class TrainROILayer : public Layer<Dtype>
{
 public:
  explicit TrainROILayer(const LayerParameter& param);
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

  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) override;
  virtual void Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) override;

 private:
  int NumROI(
      const std::vector<std::vector<cv::Rect_<Dtype> > >& roi) const;
  void MakeLabelTop(int num_roi,
                    const std::vector<std::vector<int> >& roi_label,
                    Blob<Dtype>* label_top) const;
  void MakeBBoxTop(int num_roi,
                   const std::vector<std::vector<cv::Rect_<Dtype> > >& roi_bbox,
                   Blob<Dtype>* bbox_top) const;
  cv::Size img_size_;
  cv::Size roi_size_;
  std::shared_ptr<bgm::AnnoDecoder<Dtype> > anno_decoder_;
  std::shared_ptr<bgm::TrainROIGenerator<Dtype> > roi_generator_;
  std::shared_ptr<bgm::ROIExtractor<Dtype> > roi_extractor_;
};

// inline functions
template <typename Dtype>
TrainROILayer<Dtype>::TrainROILayer(const LayerParameter& param) 
  : Layer<Dtype>(param) {

}

template <typename Dtype>
inline const char* TrainROILayer<Dtype>::type() const {
  return "TrainROI";
}

template <typename Dtype>
inline int TrainROILayer<Dtype>::ExactNumBottomBlobs() const {
  return 3;
}

template <typename Dtype>
inline int TrainROILayer<Dtype>::ExactNumTopBlobs() const {
  return 3;
}

template <typename Dtype>
inline void TrainROILayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void TrainROILayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void TrainROILayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline int TrainROILayer<Dtype>::NumROI(
    const std::vector<std::vector<cv::Rect_<Dtype> > >& roi) const {
  int num = 0;
  for (auto i = roi.cbegin(); i != roi.cend(); ++i)
    num += i->size();
  return num;
}
} // namespace caffe

#endif // !TLR_TRAIN_ROI_LAYER_HPP_
