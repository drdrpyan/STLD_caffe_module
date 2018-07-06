#ifndef BGM_YOLO_V2_HANDLER_HPP_
#define BGM_YOLO_V2_HANDLER_HPP_

#include <opencv2/core.hpp>

#include <glog/logging.h>

#include <vector>

namespace bgm
{

template <typename Dtype>
class YOLOV2Handler
{
 public:
  enum {NUM_ANCHOR_ELEM = 5};
  enum AnchorChannel {X = 0, Y, W, H, CONF, CLASS_BEGIN};

  YOLOV2Handler(const cv::Size cell_size_, int num_class,
                const std::vector<cv::Rect_<Dtype> >& anchor);
  cv::Rect_<Dtype> RawBoxToYOLOBox(const cv::Rect_<Dtype>& raw_box,
                                   int h, int w, int anchor_idx) const;
  cv::Rect_<Dtype> RawBoxToYOLOBox(const cv::Rect_<Dtype>& raw_box,
                                   const cv::Rect_<Dtype>& anchor) const;
  cv::Rect_<Dtype> YOLOBoxToRawBox(const cv::Rect_<Dtype>& yolo_box,
                                   int h, int w, int anchor_idx,
                                   bool shift = true) const;
  cv::Rect_<Dtype> YOLOBoxToRawBox(const cv::Rect_<Dtype>& yolo_box,
                                   const cv::Rect_<Dtype>& anchor,
                                   bool shift = true) const;
  int GetAnchorChannel(int anchor, AnchorChannel ch) const;
  int GetClassChannel(int anchor, int class_label) const;
  
  int NumChannels() const;
  int NumAnchorChannels() const;

  const std::vector<cv::Rect_<Dtype> >& anchor() const;
  const cv::Rect_<Dtype>& anchor(int idx) const;
  const cv::Size& cell_size() const;
  int num_class() const;

 private:
  Dtype Sigmoid(Dtype value) const;

  //cv::Size yolo_map_size_;
  cv::Size cell_size_;
  int num_class_;
  std::vector<cv::Rect_<Dtype> > anchor_;
}; // class YOLOV2Handler

// template functions
template <typename Dtype>
YOLOV2Handler<Dtype>::YOLOV2Handler(
    const cv::Size cell_size, int num_class,
    const std::vector<cv::Rect_<Dtype> >& anchor) 
  : cell_size_(cell_size), num_class_(num_class), anchor_(anchor) {
  CHECK_GT(cell_size.width, 0);
  CHECK_GT(cell_size.height, 0);
  CHECK_GT(num_class, 0);
  CHECK_GT(anchor.size(), 0);
}

template <typename Dtype>
inline cv::Rect_<Dtype> YOLOV2Handler<Dtype>::RawBoxToYOLOBox(
    const cv::Rect_<Dtype>& raw_box, int h, int w, int anchor_idx) const {
  cv::Rect_<Dtype> anchor = anchor_[anchor_idx];
  anchor.x += cell_size_.width * w;
  anchor.y += cell_size_.height * h;
  return RawBoxToYOLOBox(raw_box, anchor);
}

template <typename Dtype>
cv::Rect_<Dtype> YOLOV2Handler<Dtype>::RawBoxToYOLOBox(
    const cv::Rect_<Dtype>& raw_box, const cv::Rect_<Dtype>& anchor) const {
  cv::Rect_<Dtype> yolo_box;
  yolo_box.x = (raw_box.x + (raw_box.width / 2.) - anchor.x) / anchor.width;
  yolo_box.y = (raw_box.y + (raw_box.height / 2.) - anchor.y) / anchor.height;
  yolo_box.width = std::log(raw_box.width / anchor.width);
  yolo_box.height = std::log(raw_box.height / anchor.height);

  return yolo_box;
}

template <typename Dtype>
inline cv::Rect_<Dtype> YOLOV2Handler<Dtype>::YOLOBoxToRawBox(
    const cv::Rect_<Dtype>& yolo_box,
    int h, int w, int anchor_idx, bool shift) const {
  cv::Rect_<Dtype> anchor = anchor_[anchor_idx];
  anchor.x += cell_size_.width * w;
  anchor.y += cell_size_.height * h;
  return YOLOBoxToRawBox(yolo_box, anchor, shift);
}

template <typename Dtype>
cv::Rect_<Dtype> YOLOV2Handler<Dtype>::YOLOBoxToRawBox(
    const cv::Rect_<Dtype>& yolo_box, const cv::Rect_<Dtype>& anchor,
    bool shift) const {
  cv::Rect_<Dtype> raw_box;
  raw_box.width = std::exp(yolo_box.width) * anchor.width;
  raw_box.height = std::exp(yolo_box.height) * anchor.height;
  raw_box.x = (Sigmoid(yolo_box.x) * anchor.width) - (raw_box.width / 2.);
  raw_box.y = (Sigmoid(yolo_box.y) * anchor.height) - (raw_box.height / 2.);
  if (shift) {
    raw_box.x += anchor.x;
    raw_box.y += anchor.y;
  }

  return raw_box;
}

template <typename Dtype>
inline int YOLOV2Handler<Dtype>::GetAnchorChannel(int anchor, 
                                                  AnchorChannel ch) const {
  CHECK_GE(anchor, 0);
  CHECK_LT(anchor, anchor_.size());
  return anchor * (NUM_ANCHOR_ELEM + num_class_) + static_cast<int>(ch);
}

template <typename Dtype>
inline int YOLOV2Handler<Dtype>::GetClassChannel(
    int anchor, int class_label) const {
  CHECK_GE(anchor, 0);
  CHECK_LT(anchor, anchor_.size());
  CHECK_GE(class_label, 1);
  CHECK_LE(class_label, num_class_);
  return anchor * (NUM_ANCHOR_ELEM + num_class_) + NUM_ANCHOR_ELEM + (class_label - 1);
}

template <typename Dtype>
inline int YOLOV2Handler<Dtype>::NumChannels() const {
  return NumAnchorChannels() * anchor_.size();
}

template <typename Dtype>
inline int YOLOV2Handler<Dtype>::NumAnchorChannels() const {
  return NUM_ANCHOR_ELEM + num_class_;
}

template <typename Dtype>
inline const std::vector<cv::Rect_<Dtype> >& YOLOV2Handler<Dtype>::anchor() const {
  return anchor_;
}

template <typename Dtype>
inline const cv::Rect_<Dtype>& YOLOV2Handler<Dtype>::anchor(int idx) const {
  return anchor_[idx];
}

template <typename Dtype>
inline const cv::Size& YOLOV2Handler<Dtype>::cell_size() const {
  return cell_size_;
}

template <typename Dtype>
inline int YOLOV2Handler<Dtype>::num_class() const {
  return num_class_;
}

template <typename Dtype>
inline Dtype YOLOV2Handler<Dtype>::Sigmoid(Dtype value) const {
  return 1. / (1. + std::exp(-value));
}

} // namespace bgm
#endif // !BGM_YOLO_V2_HANDLER_HPP_
