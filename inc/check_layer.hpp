#ifndef TLR_CHECK_LAYER_HPP_
#define TLR_CHECK_LAYER_HPP_

#include "caffe/layer.hpp"

#include <opencv2/core.hpp>

namespace caffe
{

template <typename Dtype>
class CheckLayer : public Layer<Dtype>
{
 public:
  explicit CheckLayer(const LayerParameter& param);
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int MinBottomBlobs() const override;
  virtual int MaxBottomBlobs() const override;
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
  void DecodeImg(Blob<Dtype>& img_blob, 
                 std::vector<cv::Mat>* img_mat) const;
  //void DecodeLabel(const Blob<Dtype>& label_blob,
  //                std::vector<int>* label_int) const;
  //void DecodeBBox(const Blob<Dtype>& bbox_blob,
  //                std::vector<cv::Rect2f>* bbox_rect) const;
  //void DecodeBBox(const Blob<Dtype>& bbox_blob,
  //                int img_width, int img_height,
  //                std::vector<cv::Rect2f>* bbox_rect) const;
  void DecodeLabelBBox(const Blob<Dtype>& label_blob,
                       const Blob<Dtype>& bbox_blob,
                       std::vector<std::vector<int> >* label,
                       std::vector<std::vector<cv::Rect_<Dtype> > >* bbox) const;
  void DrawLabel(int label, cv::Mat& dst) const;
  void DrawGT(int label, const cv::Rect2f& rect, cv::Mat& dst) const;

  std::string OutName(const std::string& folder,
                      int label = -1, cv::Rect2f bbox = cv::Rect2f());

  //std::string dst_path_;
  bool bbox_norm_;
  float w_divider_;
  float h_divider_;

  int cnt_;
}; // class CheckLayer

// inline functions

template <typename Dtype>
inline CheckLayer<Dtype>::CheckLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {

}

template <typename Dtype>
inline const char* CheckLayer<Dtype>::type() const {
  return "Check";
}

template <typename Dtype>
inline int CheckLayer<Dtype>::MinBottomBlobs() const {
  return 1;
}

template <typename Dtype>
inline int CheckLayer<Dtype>::MaxBottomBlobs() const {
  return 3;
}

template <typename Dtype>
inline int CheckLayer<Dtype>::ExactNumTopBlobs() const {
  return 0;
}

template <typename Dtype>
inline void CheckLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void CheckLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void CheckLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

} // namespace caffe
#endif // !TLR_CHECK_LAYER_HPP_
