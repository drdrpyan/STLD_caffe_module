#include "img_augmentation.hpp"

#include "glog/logging.h"

#include "caffe/util/io.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <random>

namespace caffe
{

ImgAugmentation::ImgAugmentation() 
  : do_pad_(false), do_random_resize_(false), do_gaussian_noise_(false),
    do_mirror_(false), do_blur_(false) {

}

void ImgAugmentation::Init(const AugmentationParameter& param) {

  std::uniform_real_distribution<float> dist(0, 1);
  ////auto random_generator = std::bind(dist, random_engine_);
  random_generator_ = 
      std::bind(dist, 
                std::default_random_engine(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()));

  if (param.has_pad_param()) {
    do_pad_ = true;

    const PaddingParameter& pad_param = param.pad_param();
    pad_type_ = pad_param.type();
    pad_up_ = pad_param.pad_up();
    pad_down_ = pad_param.pad_down();
    pad_left_ = pad_param.pad_left();
    pad_right_ = pad_param.pad_right();

    CHECK_GE(pad_up_, 0);
    CHECK_GE(pad_down_, 0);
    CHECK_GE(pad_left_, 0);
    CHECK_GE(pad_right_, 0);
  }

  if (param.has_rand_resize_param()) {
    do_random_resize_ = true;

    const RandomResizeParameter& rand_resize_param = param.rand_resize_param();
    resize_prob_ = rand_resize_param.p();
    resize_w_min_ = rand_resize_param.w_min();
    resize_w_max_ = rand_resize_param.w_max();
    resize_h_min_ = rand_resize_param.h_min();
    resize_h_max_ = rand_resize_param.h_max();

    CHECK_GT(resize_prob_, 0);
    CHECK_LE(resize_prob_, 1);
    CHECK_GT(resize_w_min_, 0);
    CHECK_GT(resize_w_max_, 0);
    CHECK_GT(resize_h_min_, 0);
    CHECK_GT(resize_h_max_, 0);
  }

  if (param.gauss_noise_param().size() > 0) {
    const int NUM_NOISE = param.gauss_noise_param().size();
    do_gaussian_noise_ = true;
    noise_prob_.resize(NUM_NOISE);
    noise_mean_.resize(NUM_NOISE);
    noise_stddev_.resize(NUM_NOISE);

    for (int i = 0; i < NUM_NOISE; ++i) {
      noise_prob_[i] = param.gauss_noise_param().Get(i).p();
      noise_mean_[i] = param.gauss_noise_param().Get(i).mean();
      noise_stddev_[i] = param.gauss_noise_param().Get(i).stddev();

      CHECK_GT(noise_prob_[i], 0);
      CHECK_LE(noise_prob_[i], 1);
    }
  }

  if (param.has_mirror_param()) {
    do_mirror_ = true;
    mirror_prob_ = param.mirror_param().p();

    CHECK_GT(mirror_prob_, 0);
    CHECK_LE(mirror_prob_, 1);
  }

  if (param.has_blur_param()) {
    do_blur_ = true;
    blur_prob_ = param.blur_param().p();
    blur_kernel_ = cv::Size(param.blur_param().kernel_size(),
                            param.blur_param().kernel_size());
    blur_sigma_ = param.blur_param().sigma();

    CHECK_GT(blur_prob_, 0);
    CHECK_LE(blur_prob_, 1);
  }
}

void ImgAugmentation::Transform(
    const std::vector<ImgBBoxAnnoDatum>& src,
    std::vector<cv::Mat>* img,
    std::vector<std::vector<int> >* label,
    std::vector<std::vector<cv::Rect2f> >* bbox) {
  CHECK(img);
  CHECK(label);
  CHECK(bbox);

  img->resize(src.size());
  label->resize(src.size());
  bbox->resize(src.size());

  cv::Size src_size;

  for (int i = 0; i < src.size(); ++i) {
    cv::Mat img_mat = DecodeDatumToCVMatNative(src[i].img_datum());
    std::vector<int> temp_label(src[i].label().data(), 
                           src[i].label().data() + src[i].label_size());
    std::vector<float> x_min(src[i].x_min().data(),
                             src[i].x_min().data() + src[i].x_min_size());
    std::vector<float> x_max(src[i].x_max().data(),
                             src[i].x_max().data() + src[i].x_max_size());
    std::vector<float> y_min(src[i].y_min().data(),
                             src[i].y_min().data() + src[i].y_min_size());
    std::vector<float> y_max(src[i].y_max().data(),
                             src[i].y_max().data() + src[i].y_max_size());
    CHECK_EQ(temp_label.size(), x_min.size());
    CHECK_EQ(x_min.size(), x_max.size());
    CHECK_EQ(x_max.size(), y_min.size());
    CHECK_EQ(y_min.size(), y_max.size());

    std::vector<cv::Rect2f> temp_bbox(temp_label.size());
    for (int j = 0; j < temp_bbox.size(); ++j)
      temp_bbox[j] = cv::Rect2f(x_min[j], y_min[j], 
                                x_max[j] - x_min[j], 
                                y_max[j] - y_min[j]);

    if (i == 0) {
      img_size_ = img_mat.size();
      src_size = img_mat.size();
    }
    else
      CHECK(src_size == img_mat.size());

    if (do_random_resize_)
      RandomResize(img_mat, temp_bbox, i==0);
    if (do_pad_)
      Pad(img_mat, temp_bbox);
    if (do_mirror_)
      Mirror(img_mat, temp_bbox);
    if (do_gaussian_noise_)
      GaussianNoise(img_mat);
    if (do_blur_)
      Blur(img_mat);

    (*img)[i] = img_mat;
    (*label)[i] = temp_label;
    (*bbox)[i] = temp_bbox;
  }
}

void ImgAugmentation::Pad(cv::Mat& img,
                          std::vector<cv::Rect2f>& bbox) {
  switch (pad_type_) {
    case PaddingParameter::ZERO:
      cv::copyMakeBorder(img, img, pad_up_, pad_down_, pad_left_, pad_right_, IPL_BORDER_CONSTANT);
      break;
    case PaddingParameter::MIRROR:
      cv::copyMakeBorder(img, img, pad_up_, pad_down_, pad_left_, pad_right_, IPL_BORDER_REPLICATE);
      break;
    default:
      LOG(FATAL) << "Illegal padding type";
   }

  for (auto iter = bbox.begin(); iter != bbox.end(); ++iter) {
    iter->x += pad_left_;
    iter->y += pad_up_;
  }
}

void ImgAugmentation::RandomResize(cv::Mat& img,
                                   std::vector<cv::Rect2f>& bbox,
                                   bool new_size) {
  float w_scale, h_scale;

  if (new_size && GetRandom() <= resize_prob_) {
    w_scale = GetRandom(resize_w_min_, resize_w_max_);
    h_scale = GetRandom(resize_h_min_, resize_h_max_);

    img_size_ = cv::Size(img.cols * w_scale, img.rows * h_scale);
  }
  else {
    w_scale = img_size_.width / static_cast<float>(img.cols);
    h_scale = img_size_.height / static_cast<float>(img.rows);
  }

  cv::resize(img, img, img_size_);

  for (auto iter = bbox.begin(); iter != bbox.end(); ++iter) {
    iter->x *= w_scale;
    iter->y *= h_scale;
    iter->width *= w_scale;
    iter->height *= h_scale;
  }
}

void ImgAugmentation::GaussianNoise(cv::Mat& img) {
  int idx = std::round(GetRandom(0, noise_prob_.size() - 1));
  if (GetRandom() < noise_prob_[idx]) {
    cv::Mat noised;

    if (img.type() == CV_8UC3)
      img.convertTo(noised, CV_32FC3, 1 / 255.0);
    else
      LOG(FATAL) << "Not implemented yet";

    cv::Mat noise(img.size(), CV_32FC3);
    cv::randn(noise, noise_mean_[idx], noise_stddev_[idx]);

    noised += noise;
    cv::normalize(noised, noised, 0.0, 1.0, CV_MINMAX, CV_32F);

    noised.convertTo(img, CV_8UC3, 255);    
  }
}

void ImgAugmentation::Mirror(cv::Mat& img,
                             std::vector<cv::Rect2f>& bbox) {
  if (GetRandom() < mirror_prob_) {
    cv::flip(img, img, 1);

    float center = img.cols / 2.;
    for (auto iter = bbox.begin(); iter != bbox.end(); ++iter) {
      iter->x = 2 * center - (iter->x + iter->width);
    }
  }
}

void ImgAugmentation::Blur(cv::Mat& img) {
  if (GetRandom() < blur_prob_) {
    cv::GaussianBlur(img, img, blur_kernel_, blur_sigma_);
  }
}

} // namespace caffe