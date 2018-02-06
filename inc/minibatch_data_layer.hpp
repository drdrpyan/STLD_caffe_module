#ifndef TLR_MINIBATCH_DATA_LAYER_HPP_
#define TLR_MINIBATCH_DATA_LAYER_HPP_

#include "base_img_bbox_data_layer.hpp"

#include "obj_contained.hpp"
#include "recycling_queue.hpp"

#include <functional>

namespace caffe
{

template <typename Dtype>
class MinibatchDataLayer : public BaseImgBBoxDataLayer<Dtype>
{
  struct Minibatch
  {
    cv::Rect roi;
    std::vector<int> label;
    std::vector<cv::Rect_<Dtype> > bbox;
  };

  struct MinibatchBlob
  {
    Blob<Dtype> data;
    Blob<Dtype> label;
    Blob<Dtype> bbox;
  };

 public:
  explicit MinibatchDataLayer(const LayerParameter& param);
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
  void SelectMinibatch(int src_width, int src_height,
                       const std::vector<int>& src_label,
                       const std::vector<cv::Rect_<Dtype> >& src_bbox,
                       std::vector<Minibatch>* minibatch);
  void MakeDataBlob(const Blob<Dtype>& src_data, int batch_idx,
                    const cv::Rect& roi, Blob<Dtype>* dst_data) const;
  void MakeLabelBlob(const std::vector<int> label,
                     Blob<Dtype>* dst_label) const;
  void MakeBBoxBlob(const std::vector<cv::Rect_<Dtype> >& bbox,
                    Blob<Dtype>* dst_bbox) const;
  ////void MakeDataBlob(const Blob<Dtype>& src_data,
  ////                  const std::vector<std::vector<Minibatch> >& minibatch,
  ////                  Blob<Dtype>* dst_data) const;
  ////void MakeLabelBlob(const std::vector<std::vector<Minibatch> >& minibatch,
  ////                   Blob<Dtype>* dst_label) const;
  ////void MakeBBoxBlob(const std::vector<std::vector<Minibatch> >& minibatch,
  ////                  Blob<Dtype>* dst_bbox) const;
  void MakeROIBlob(const std::vector<std::vector<Minibatch> >& minibatch,
                   Blob<Dtype>* dst_roi) const;
  void ExtractMinibatchBlob(bool make_label_blob = true,
                            bool make_bbox_blob = true);
  void MakeTopBlob(const vector<Blob<Dtype>*>& top);

  int Random(int min, int max);

  int num_batch_;
  int max_num_patch_;
  int num_gt_;
  int width_;
  int height_;

  std::function<int(void)> rng_;

  std::unique_ptr<bgm::ObjContained<Dtype> > obj_contained_;

  bgm::RecyclingQueue<MinibatchBlob> minibatch_queue_;
  
}; // class MinibatchDataLayer

// inline functions
template <typename Dtype>
inline MinibatchDataLayer<Dtype>::MinibatchDataLayer(
    const LayerParameter& param) : BaseImgBBoxDataLayer<Dtype>(param) {

}

template <typename Dtype>
inline const char* MinibatchDataLayer<Dtype>::type() const {
  return "MinibatchData";
}

template <typename Dtype>
inline int MinibatchDataLayer<Dtype>::MinTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline int MinibatchDataLayer<Dtype>::MaxTopBlobs() const {
  return 3;
}

template <typename Dtype>
inline void MinibatchDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
                 const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline int MinibatchDataLayer<Dtype>::Random(int min, int max) {
  CHECK_LE(min, max);

  return (min == max) ? min : (rng_() % (max - min)) + min;
}
} // namespace caffe
#endif // !TLR_MINIBATCH_DATA_LAYER_HPP_