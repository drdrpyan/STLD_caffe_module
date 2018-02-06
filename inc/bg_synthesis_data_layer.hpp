#ifndef TLR_BG_SYNTHESIS_DATA_LAYER_HPP_
#define TLR_BG_SYNTHESIS_DATA_LAYER_HPP_

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"

#include <opencv2/core.hpp>

#include <list>
#include <random>

namespace caffe
{

template <typename Dtype> 
class BGSynthesisDataLayer : public BaseDataLayer<Dtype> 
{
  enum {NUM_OBJ_LOAD = 10,
        NUM_BG_LOAD = 10,
        NUM_SYNTH_TRY = 5};
 public:
  BGSynthesisDataLayer(LayerParameter param);
  virtual void DataLayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;

 private:
  void LoadObj(int num = NUM_OBJ_LOAD);
  void LoadBG(int num = NUM_BG_LOAD);
  
  void CropBG(const cv::Mat& src,
              int num, std::vector<cv::Mat>* bg);
  void Synthesis(const std::vector<cv::Mat>& obj, 
                 cv::Mat* bg,
                 std::vector<cv::Rect>* location) const;
  void GetSynthROI(const cv::Rect& roi,
                   cv::Rect* obj_roi, 
                   cv::Rect* bg_roi) const;
  bool Overlap(const std::vector<cv::Rect>& prev_roi,
               const cv::Rect& new_roi) const;
  void MakeTop(const std::vector<cv::Mat>& data,
               int num_nonneg,
               const std::vector<std::vector<int> >& label,
               const std::vector<std::vector<cv::Rect> >& bbox,
               const std::vector<Blob<Dtype>*>& top);
  void MatToBlob(const std::vector<cv::Mat>& mat,
                 Dtype* blob_data) const;
  void BlobToMat(const Blob<Dtype>& blob,
                 std::vector<cv::Mat>* mat) const;
  void GetUniformRandomInteger(int num, int min, int max,
                               std::vector<int>* random) const;
  void GetUniformRandomReal(int num, float min, float max,
                            std::vector<float>* random) const;
  bool TryBernoulli(float p);
  cv::Mat RandomResize(const cv::Mat& src, 
                       float scale_min, float scale_max,
                       int scale_precision = 1) const;
  cv::Mat ApplyGaussianNoise(const cv::Mat src) const;
  //cv::Mat ApplyPepperSaltNoise(const cv::Mat src) const;

  std::pair<cv::Mat, int> PopObj();
  cv::Mat PopBG();

  int width_;
  int height_;
  int max_obj_;
  int min_obj_;
  int batch_size_;
  int num_neg_;

  cv::Rect activation_region_;
  ActivationRegionParameter::ActivationMethod activation_method_;

  std::shared_ptr<ImageDataLayer<Dtype> > obj_data_layer_;
  Blob<Dtype> obj_data_, obj_label_;
  std::vector<Blob<Dtype>*> obj_data_out_;

  std::shared_ptr<ImageDataLayer<Dtype> > bg_data_layer_;
  Blob<Dtype> bg_data_, bg_label_;
  std::vector<Blob<Dtype>*> bg_data_out_;
  
  std::list<std::pair<cv::Mat, int> > obj_;
  std::list<cv::Mat> bg_;

  std::mt19937 random_engine_;
  //std::shared_ptr<std::uniform_int_distribution<int> > uniform_dist_;
  //std::function<int(void)> random_generator_;

  std::list<float> random_prob_;

  float resize_prob_;
  float resize_min_, resize_max_;
  float gaussian_noise_prob_;
};

// inline functions
template <typename Dtype> 
inline BGSynthesisDataLayer<Dtype>::BGSynthesisDataLayer(
    LayerParameter param) 
  : BaseDataLayer<Dtype>(param),
    random_engine_(std::random_device()()) {

}

template <typename Dtype> 
inline const char* BGSynthesisDataLayer<Dtype>::type() const {
  return "BGSynthesisData";
}

template <typename Dtype> 
inline void BGSynthesisDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

} // namespace caffe
#endif // !TLR_BG_SYNTHESIS_DATA_LAYER_HPP_