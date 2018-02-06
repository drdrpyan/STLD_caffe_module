#ifndef BGM_DETECTION_UTIL_HPP_
#define BGM_DETECTION_UTIL_HPP_

#include <opencv2/core.hpp>

#include <glog/logging.h>

#include <list>
#include <numeric>
#include <cmath>

namespace bgm
{

template <typename Dtype>
struct BoxAnnotation
{
  int label;
  cv::Rect_<Dtype> bbox;
};

template <typename Dtype>
struct Detection
{
  int label;
  cv::Rect_<Dtype> bbox;
  float conf;
};

typedef Detection<int> DetectionI;
typedef Detection<float> DetectionF;
typedef Detection<double> DetectionD;

template <typename Dtype>
float CalcIoU(const cv::Rect_<Dtype>& box1, const cv::Rect_<Dtype>& box2) {
  Dtype x_min = std::max(box1.x, box2.x);
  Dtype y_min = std::max(box1.y, box2.y);
  Dtype x_max = std::min(box1.x + box1.width, box2.x + box2.width);
  Dtype y_max = std::min(box1.y + box1.height, box2.y + box2.height);

  Dtype inter_w = std::max(x_max - x_min, static_cast<Dtype>(0));
  Dtype inter_h = std::max(y_max - y_min, static_cast<Dtype>(0));
  Dtype inter_area = inter_w * inter_h;

  Dtype union_area = box1.area() + box2.area() - inter_area;

  return inter_area / static_cast<float>(union_area);
}

template <typename Dtype>
void DetectionMatching(const std::vector<Detection<Dtype> >& detection,
                       const std::vector<BoxAnnotation<Dtype> >& gt,
                       float conf_threshold, float iou_threshold,
                       std::vector<int>* tp_idx, std::vector<int>* fp_idx,
                       std::vector<int>* fn_idx) {
  CHECK(tp_idx);
  CHECK(fp_idx);
  CHECK(fn_idx);
  tp_idx->clear();
  fp_idx->clear();
  fn_idx->clear();

  std::list<int> not_matched(detection.size());
  std::iota(not_matched.begin(), not_matched.end(), 0);

  auto fp_iter = not_matched.begin();
  while (fp_iter != not_matched.end()) {
    if (detection[*fp_iter].conf < conf_threshold) {
      fp_idx->push_back(*fp_iter);
      not_matched.erase(fp_iter++);
    }
    else
      ++fp_iter;
  }

  for (int i = 0; i < gt.size(); ++i) {
    auto max_idx_iter = not_matched.end();
    float max_iou = 0;
    for (auto not_match_idx_iter = not_matched.begin();
         not_match_idx_iter != not_matched.end();
         ++not_match_idx_iter) {
      float iou = CalcIoU(gt[i].bbox,
                          detection[*not_match_idx_iter].bbox);
      if (iou > max_iou) {
        max_idx_iter = not_match_idx_iter;
        max_iou = iou;
      }
    }

    if (max_iou >= iou_threshold && max_idx_iter != not_matched.end()) {
      tp_idx->push_back(*max_idx_iter);
      not_matched.erase(max_idx_iter);
    }
    else
      fn_idx->push_back(i);
  }

  fp_idx->insert(fp_idx->end(), not_matched.begin(), not_matched.end());
}

template <typename Dtype>
Dtype Sigmoid(Dtype value) {
  return 1. / (1 + std::exp(-value));
}


} // namespace bgm

#endif // !BGM_DETECTION_UTIL_HPP_
