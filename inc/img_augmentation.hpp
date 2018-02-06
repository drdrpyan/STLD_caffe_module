#ifndef TLR_IMG_AUGMENTATION_HPP_
#define TLR_IMG_AUGMENTATION_HPP_

#include "caffe/proto/caffe.pb.h"

#include <opencv2/core.hpp>

#include <functional>

namespace caffe
{

class ImgAugmentation
{
 public:
  ImgAugmentation();
  void Init(const AugmentationParameter& param);
  void Transform(const std::vector<ImgBBoxAnnoDatum>& src,
                 std::vector<cv::Mat>* img,
                 std::vector<std::vector<int> >* label,
                 std::vector<std::vector<cv::Rect2f> >* bbox);
  const cv::Size& img_size() const;

 private:
  float GetRandom();
  float GetRandom(float min, float max);

  void Pad(cv::Mat& img,
           std::vector<cv::Rect2f>& bbox);
  void RandomResize(cv::Mat& img,
                    std::vector<cv::Rect2f>& bbox,
                    bool new_size = true);
  void GaussianNoise(cv::Mat& img);
  void Mirror(cv::Mat& img,
              std::vector<cv::Rect2f>& bbox);
  void Blur(cv::Mat& img);
  
  std::function<float(void)> random_generator_;

  cv::Size img_size_;

  bool do_pad_;
  PaddingParameter::PaddingType pad_type_;
  int pad_up_;
  int pad_down_;
  int pad_left_;
  int pad_right_;

  bool do_random_resize_;
  float resize_prob_;
  float resize_w_min_;
  float resize_w_max_;
  float resize_h_min_;
  float resize_h_max_;

  bool do_gaussian_noise_;
  std::vector<float> noise_prob_;
  std::vector<float> noise_mean_;
  std::vector<float> noise_stddev_;

  bool do_mirror_;
  float mirror_prob_;

  bool do_blur_;
  float blur_prob_;
  cv::Size blur_kernel_;
  float blur_sigma_;
};

// inline functions
inline const cv::Size& ImgAugmentation::img_size() const {
  if (do_pad_)
    return cv::Size(img_size_.width + pad_left_ + pad_right_,
                    img_size_.height + pad_up_ + pad_down_);
  else
    return img_size_;
}

inline float ImgAugmentation::GetRandom() {
  return random_generator_();
}

inline float ImgAugmentation::GetRandom(float min, float max) {
  return GetRandom() * (max - min) + min;
}

} // namespace caffe
#endif // !TLR_IMG_AUGMENTATION_HPP_