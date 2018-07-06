#ifndef BGM_DHA_HANDLER_HPP_
#define BGM_DHA_HANDLER_HPP_

//#include "tlr_math.hpp"
#include "detection_util.hpp"

#include <opencv2/core.hpp>

#include <glog/logging.h>

#include <vector>

namespace bgm
{

template <typename Dtype>
class DHAHandler
{
  public:
  enum AnchorElem { X = 0, Y, W, H, CLASS_BEGIN };

  DHAHandler(const cv::Size cell_size_, int num_class,
             const std::vector<cv::Rect_<Dtype> >& anchor);
  cv::Rect_<Dtype> RawBoxToNormalizedBox(
    const cv::Rect_<Dtype>& raw_box, int h, int w, int anchor_idx) const;
  cv::Rect_<Dtype> RawBoxToNormalizedBox(
    const cv::Rect_<Dtype>& raw_box, const cv::Rect_<Dtype>& anchor) const;
  cv::Rect_<Dtype> NormalizedBoxToRawBox(
    const cv::Rect_<Dtype>& normalized_box,
    int h, int w, int anchor_idx, bool shift = true) const;
  cv::Rect_<Dtype> NormalizedBoxToRawBox(
    const cv::Rect_<Dtype>& normalized_box,
    const cv::Rect_<Dtype>& anchor, bool shift = true) const;

  int GetPosAnchorConfChannel(int anchor) const;
  int GetNegAnchorConfChannel() const;
  int GetAnchorChannel(int anchor, AnchorElem ch) const;
  int GetClassChannel(int anchor, int class_label,
                      bool zero_begin = true) const;

  const std::vector<cv::Rect_<Dtype> >& anchor() const;
  const cv::Rect_<Dtype>& anchor(int idx) const;
  const cv::Size& cell_size() const;
  int num_class() const;

  private:
  cv::Size cell_size_;
  int num_class_;
  std::vector<cv::Rect_<Dtype> > anchor_;
}; // class DHAHandler

// fucntion definition
template <typename Dtype>
DHAHandler<Dtype>::DHAHandler(
  const cv::Size cell_size, int num_class,
  const std::vector<cv::Rect_<Dtype> >& anchor)
  : cell_size_(cell_size), num_class_(num_class), anchor_(anchor) {
  CHECK_GT(cell_size.width, 0);
  CHECK_GT(cell_size.height, 0);
  CHECK_GT(num_class, 0);
  CHECK_GT(anchor.size(), 0);
}

template <typename Dtype>
inline cv::Rect_<Dtype> DHAHandler<Dtype>::RawBoxToNormalizedBox(
  const cv::Rect_<Dtype>& raw_box, int h, int w, int anchor_idx) const {
  cv::Rect_<Dtype> anchor = anchor_[anchor_idx];
  anchor.x += cell_size_.width * w;
  anchor.y += cell_size_.height * h;
  return RawBoxToNormalizedBox(raw_box, anchor);
}

template <typename Dtype>
cv::Rect_<Dtype> DHAHandler<Dtype>::RawBoxToNormalizedBox(
  const cv::Rect_<Dtype>& raw_box, const cv::Rect_<Dtype>& anchor) const {
  cv::Rect_<Dtype> normalized_box;
  normalized_box.x = (raw_box.x + (raw_box.width / 2.) - anchor.x) / anchor.width;
  normalized_box.y = (raw_box.y + (raw_box.height / 2.) - anchor.y) / anchor.height;
  normalized_box.width = std::log(raw_box.width / anchor.width);
  normalized_box.height = std::log(raw_box.height / anchor.height);

  return normalized_box;
}

template <typename Dtype>
inline cv::Rect_<Dtype> DHAHandler<Dtype>::NormalizedBoxToRawBox(
  const cv::Rect_<Dtype>& normalized_box,
  int h, int w, int anchor_idx, bool shift) const {
  cv::Rect_<Dtype> anchor = anchor_[anchor_idx];
  anchor.x += cell_size_.width * w;
  anchor.y += cell_size_.height * h;
  return NormalizedBoxToRawBox(normalized_box, anchor, shift);
}

template <typename Dtype>
cv::Rect_<Dtype> DHAHandler<Dtype>::NormalizedBoxToRawBox(
  const cv::Rect_<Dtype>& normalized_box,
  const cv::Rect_<Dtype>& anchor, bool shift) const {
  cv::Rect_<Dtype> raw_box;
  raw_box.width = std::exp(normalized_box.width) * anchor.width;
  raw_box.height = std::exp(normalized_box.height) * anchor.height;
  raw_box.x = (Sigmoid(normalized_box.x) * anchor.width) - (raw_box.width / 2.);
  raw_box.y = (Sigmoid(normalized_box.y) * anchor.height) - (raw_box.height / 2.);
  if (shift) {
    raw_box.x += anchor.x;
    raw_box.y += anchor.y;
  }

  return raw_box;
}

template <typename Dtype>
inline int DHAHandler<Dtype>::GetPosAnchorConfChannel(int anchor) const {
  CHECK_GE(anchor, 0);
  CHECK_LT(anchor, anchor_.size());
  return anchor;
}

template <typename Dtype>
inline int DHAHandler<Dtype>::GetNegAnchorConfChannel() const {
  return anchor_.size();
}

template <typename Dtype>
inline int DHAHandler<Dtype>::GetAnchorChannel(int anchor, AnchorElem ch) const {
  CHECK_GE(anchor, 0);
  CHECK_LT(anchor, anchor_.size());
  return (anchor_.size() + 1) + (4 + num_class_)*anchor + static_cast<int>(ch);
}

template <typename Dtype>
inline int DHAHandler<Dtype>::GetClassChannel(int anchor, int class_label,
                                              bool zero_begin) const {
  int label = zero_begin ? class_label : class_label - 1;
  CHECK_GE(label, 0);
  CHECK_LT(label, num_class_);
  return GetAnchorChannel(anchor, AnchorElem::CLASS_BEGIN) + label;
}



template <typename Dtype>
inline const std::vector<cv::Rect_<Dtype> >&
DHAHandler<Dtype>::anchor() const {
  return anchor_;
}

template <typename Dtype>
inline const cv::Rect_<Dtype>& DHAHandler<Dtype>::anchor(int idx) const {
  return anchor_[idx];
}

template <typename Dtype>
inline const cv::Size& DHAHandler<Dtype>::cell_size() const {
  return cell_size_;
}

template <typename Dtype>
inline int DHAHandler<Dtype>::num_class() const {
  return num_class_;
}

} // namespace bgm
#endif // !BGM_DHA_HANDLER_HPP_
