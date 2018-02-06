#ifndef TLR_OBJ_CONTAINED_HPP_
#define TLR_OBJ_CONTAINED_HPP_

#include <opencv2/core.hpp>
#include <glog/logging.h>

namespace bgm
{

template <typename Dtype>
class ObjContained
{
 public:
  virtual bool operator()(const cv::Rect_<Dtype>& ground,
                          const cv::Rect_<Dtype>& obj) = 0;
};

template <typename Dtype>
class WholeObjContained : public ObjContained<Dtype>
{
 public:
  virtual bool operator()(const cv::Rect_<Dtype>& ground,
                          const cv::Rect_<Dtype>& obj) override {
    return (ground.x <= obj.x) && (ground.y <= obj.y) &&
      (ground.br().x >= obj.br().x) && (ground.br().y >= obj.br().y);
  };
}; // class WholeObjContained

template <typename Dtype>
class ObjCenterContained : public ObjContained<Dtype>
{
 public:
  virtual bool operator()(const cv::Rect_<Dtype>& ground,
                          const cv::Rect_<Dtype>& obj) override {
    Dtype center_x = obj.x + obj.width / 2.;
    Dtype center_y = obj.y + obj.height / 2.;
    return (ground.x <= center_x) && (ground.br().x >= center_x) &&
      (ground.y <= center_y) && (ground.br().y >= center_y);
  }
};

template <typename Dtype>
class IntersectionOverObjContained : public ObjContained<Dtype>
{
 public:
  explicit IntersectionOverObjContained(float threshold = 0.5f) {
    set_threshold(threshold);
  }

  void set_threshold(float threshold) {
    CHECK_GE(threshold, 0);
    CHECK_LE(threshold, 1);
    threshold_ = threshold;
  }

  virtual bool operator()(const cv::Rect_<Dtype>& ground,
                          const cv::Rect_<Dtype>& obj) override {
    //Dtype obj_area = obj.area();

    //Dtype x_overlap = Overlap(ground.x, ground.br().x,
    //                          obj.x, obj.br().x);
    //Dtype y_overlap = Overlap(ground.y, ground.br().y,
    //                          obj.y, obj.br().y);
    //Dtype intersection = x_overlap * y_overlap;

    //return (intersection / static_cast<float>(obj_area)) >= threshold_;
    return Score(ground, obj) >= threshold_;
  };

  Dtype Score(const cv::Rect_<Dtype>& ground,
              const cv::Rect_<Dtype>& obj) const {
    Dtype obj_area = obj.area();

    Dtype x_overlap = Overlap(ground.x, ground.br().x,
                              obj.x, obj.br().x);
    Dtype y_overlap = Overlap(ground.y, ground.br().y,
                              obj.y, obj.br().y);
    Dtype intersection = x_overlap * y_overlap;

    return intersection / obj_area;
  }

 private:
  Dtype Overlap(Dtype a_min, Dtype a_max,
                Dtype b_min, Dtype b_max) const {
    Dtype overlap = std::min(a_max, b_max) - std::max(a_min, b_min);
    return std::max(static_cast<Dtype>(0), overlap);
  }

  float threshold_;
};

} // namespace bgm

#endif // !TLR_OBJ_CONTAINED_HPP_
