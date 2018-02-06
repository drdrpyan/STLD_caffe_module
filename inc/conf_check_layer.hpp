#ifndef TLR_CONF_CHECK_LAYER_HPP_
#define TLR_CONF_CHECK_LAYER_HPP_

#include "caffe/layer.hpp"

#include "anno_decoder.hpp"
#include "img_decoder.hpp"
#include "obj_contained.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <fstream>

namespace caffe
{

template <typename Dtype>
class ConfCheckLayer : public Layer<Dtype>
{
 public:
  explicit ConfCheckLayer(const LayerParameter& param);
  virtual ~ConfCheckLayer() override;
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int MinBottomBlobs() const override;
  virtual int MaxBottomBlobs() const override;
  //virtual int ExactNumTopBlobs() const override;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom) override;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom) override;

 private:
  void DecodeGT(const Blob<Dtype>& gt_blob, 
                std::vector<bool>* pos_neg) const;
  std::string GetImgName(int img_cnt) const;
  cv::Mat DrawResult(const cv::Mat& src, bool gt, Dtype conf) const;
  cv::Mat DrawGlobalResult(const cv::Mat& src,
                           const std::vector<bool>& gt,
                           const std::vector<Dtype>& conf) const;

  std::string out_path_;
  bool draw_;

  cv::Size win_size_;
  std::vector<cv::Point> win_offset_;
  bool global_detection_;

  int img_cnt_;

  std::ofstream log_;
  std::unique_ptr<bgm::AnnoDecoder<Dtype> > anno_decoder_;
  std::unique_ptr<bgm::ImgDecoder<Dtype> > img_decoder_;
  std::unique_ptr<bgm::ObjContained<Dtype> > obj_contained_;
};

// inline functions
template <typename Dtype>
ConfCheckLayer<Dtype>::ConfCheckLayer(const LayerParameter& param) 
  : Layer<Dtype>(param) {

}

template <typename Dtype>
ConfCheckLayer<Dtype>::~ConfCheckLayer() {
  log_.close();
}

template <typename Dtype>
inline const char* ConfCheckLayer<Dtype>::type() const {
  return "ConfCheck";
}

template <typename Dtype>
inline int ConfCheckLayer<Dtype>::MinBottomBlobs() const {
  return 2;
}

template <typename Dtype>
inline int ConfCheckLayer<Dtype>::MaxBottomBlobs() const {
  return 3;
}

template <typename Dtype>
inline void ConfCheckLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void ConfCheckLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void ConfCheckLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline std::string ConfCheckLayer<Dtype>::GetImgName(int img_cnt) const {
  char img_name[256];
  std::sprintf(img_name, "%06d.jpg", img_cnt);
  return img_name;
}

} // namespace caffe
#endif // !TLR_CONF_CHECK_LAYER_HPP_
