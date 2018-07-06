#ifndef BGM_DETECTION_NMS_HPP_
#define BGM_DETECTION_NMS_HPP_

#include "detection_util.hpp"

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <list>
#include <numeric>

#define NEAR_DIST 3

namespace bgm
{

template <typename Dtype>
class DetectionNMS
{
 public:
  virtual void operator()(const std::vector<Detection<Dtype> >& detection,
                          std::vector<Detection<Dtype> >* result) = 0;

  private:
}; // class DetectionNMS

template <typename Dtype>
class VOCNMS : public DetectionNMS<Dtype>
{
 public:
  VOCNMS(float overlap_threshold = 0.5f);
  virtual void operator()(const std::vector<Detection<Dtype> >& detection,
                          std::vector<Detection<Dtype> >* result);

 protected:
  void CalcArea(const std::vector<Detection<Dtype> >& detection,
                  std::vector<float>* area) const;
  float overlap_threshold() const;

  void YMaxArgSort(const std::vector<Detection<Dtype> >& detection,
                   std::list<int>* idx) const;
  //void CalcArea(const std::vector<Detection<Dtype> >& detection,
  //              std::vector<float>* area) const;

 private:
  float overlap_threshold_;
}; // class DetectionNMS

// template functions
template <typename Dtype>
VOCNMS<Dtype>::VOCNMS(float overlap_threshold = 0.5f) 
  : overlap_threshold_(overlap_threshold) {

}

template <typename Dtype>
void VOCNMS<Dtype>::operator()(
    const std::vector<Detection<Dtype> >& detection,
    std::vector<Detection<Dtype> >* result) {
  CHECK(result);

  std::list<int> idx;
  YMaxArgSort(detection, &idx);

  std::vector<float> area;
  CalcArea(detection, &area);

  std::vector<int> pick_idx;
  while (idx.size() > 0) {
    int i = idx.back();
    pick_idx.push_back(i);
    idx.pop_back();

    const cv::Rect_<Dtype>& box1 = detection[i].bbox;

    auto idx_iter = idx.begin();
    while (idx_iter != idx.end()) {
      const cv::Rect_<Dtype>& box2 = detection[*idx_iter].bbox;
      Dtype x_min = std::max(box1.x, box2.x);
      Dtype y_min = std::max(box1.y, box2.y);
      Dtype x_max = std::min(box1.x + box1.width, box2.x + box2.width);
      Dtype y_max = std::min(box1.y + box1.height, box2.y + box2.height);
      
      Dtype bound_area = 
          std::max(x_max - x_min, static_cast<Dtype>(0)) * 
            std::max(y_max - y_min, static_cast<Dtype>(0));
      float overlap = bound_area / area[*idx_iter];

      if (overlap > overlap_threshold_)
        idx.erase(idx_iter++);
      else
        ++idx_iter;
    }
  }

  result->resize(pick_idx.size());
  for (int i = 0; i < pick_idx.size(); ++i)
    (*result)[i] = detection[pick_idx[i]];
}

template <typename Dtype>
void VOCNMS<Dtype>::YMaxArgSort(
    const std::vector<Detection<Dtype> >& detection,
    std::list<int>* idx) const {
  CHECK(idx);

  idx->resize(detection.size());
  std::vector<int> idx_vec(detection.size());
  std::iota(idx_vec.begin(), idx_vec.end(), 0);

  //std::sort(
  //    idx->begin(), idx->end(), 
  //    [&detection](int i1, int i2){
  //  return true;/*(detection[i1].bbox.y + detection[i1].bbox.height) > (detection[i2].bbox.y + detection[i2].bbox.height);*/ });
  std::sort(idx_vec.begin(), idx_vec.end(),
            [&detection](int i1, int i2) {
                return (detection[i1].bbox.y + detection[i1].bbox.height) < (detection[i2].bbox.y + detection[i2].bbox.height); });

  idx->assign(idx_vec.begin(), idx_vec.end());
}

template <typename Dtype>
void VOCNMS<Dtype>::CalcArea(
    const std::vector<Detection<Dtype> >& detection,
    std::vector<float>* area) const {
  CHECK(area);

  area->resize(detection.size());
  for (int i = 0; i < detection.size(); ++i)
    (*area)[i] = detection[i].bbox.area();
}

template <typename Dtype>
inline float VOCNMS<Dtype>::overlap_threshold() const {
  return overlap_threshold_;
}

template <typename Dtype>
class ConfMaxVOCNMS : public VOCNMS<Dtype>
{
 public:
  ConfMaxVOCNMS(float overlap_threshold = 0.5f);
  virtual void operator()(const std::vector<Detection<Dtype> >& detection,
                          std::vector<Detection<Dtype> >* result);
 protected:
  void ConfMaxArgSort(const std::vector<Detection<Dtype> >& detection,
                      std::list<int>* idx);
};

template <typename Dtype>
ConfMaxVOCNMS<Dtype>::ConfMaxVOCNMS(float overlap_threshold)
  : VOCNMS<Dtype>(overlap_threshold) { }

template <typename Dtype>
void ConfMaxVOCNMS<Dtype>::operator()(
    const std::vector<Detection<Dtype> >& detection,
    std::vector<Detection<Dtype> >* result) {
    CHECK(result);

  std::list<int> idx;
  ConfMaxArgSort(detection, &idx);

  std::vector<float> area;
  CalcArea(detection, &area);

  std::vector<int> pick_idx;
  while (idx.size() > 0) {
    int i = idx.back();
    pick_idx.push_back(i);
    idx.pop_back();

    const cv::Rect_<Dtype>& box1 = detection[i].bbox;

    auto idx_iter = idx.begin();
    while (idx_iter != idx.end()) {
      const cv::Rect_<Dtype>& box2 = detection[*idx_iter].bbox;
      Dtype x_min = std::max(box1.x, box2.x);
      Dtype y_min = std::max(box1.y, box2.y);
      Dtype x_max = std::min(box1.x + box1.width, box2.x + box2.width);
      Dtype y_max = std::min(box1.y + box1.height, box2.y + box2.height);
      
      Dtype bound_area = 
          std::max(x_max - x_min, static_cast<Dtype>(0)) * 
            std::max(y_max - y_min, static_cast<Dtype>(0));
      float overlap = bound_area / area[*idx_iter];

      if (overlap > overlap_threshold())
        idx.erase(idx_iter++);
      else
        ++idx_iter;
    }
  }

  result->resize(pick_idx.size());
  for (int i = 0; i < pick_idx.size(); ++i)
    (*result)[i] = detection[pick_idx[i]];
}

template <typename Dtype>
void ConfMaxVOCNMS<Dtype>::ConfMaxArgSort(
    const std::vector<Detection<Dtype> >& detection, std::list<int>* idx) {
  CHECK(idx);

  idx->resize(detection.size());
  std::vector<int> idx_vec(detection.size());
  std::iota(idx_vec.begin(), idx_vec.end(), 0);

  std::sort(idx_vec.begin(), idx_vec.end(),
                   [&detection](int i1, int i2) {
                      return (detection[i1].bbox.y + detection[i1].bbox.height) < (detection[i2].bbox.y + detection[i2].bbox.height); });

  std::stable_sort(idx_vec.begin(), idx_vec.end(),
            [&detection](int i1, int i2) {
                return (detection[i1].conf) < (detection[i2].conf); });


  idx->assign(idx_vec.begin(), idx_vec.end());
}

template <typename Dtype>
class MeanSizeNMS : public ConfMaxVOCNMS<Dtype>
{
 public:
  MeanSizeNMS(float overlap_threshold = 0.5f);
  virtual void operator()(const std::vector<Detection<Dtype> >& detection,
                          std::vector<Detection<Dtype> >* result);
 private:
};

template <typename Dtype>
MeanSizeNMS<Dtype>::MeanSizeNMS(float overlap_threshold) 
  : ConfMaxVOCNMS<Dtype>(overlap_threshold) {
}

template <typename Dtype>
void MeanSizeNMS<Dtype>::operator()(
    const std::vector<Detection<Dtype> >& detection,
    std::vector<Detection<Dtype> >* result) {
  CHECK(result);
  result->clear();

  std::list<int> idx;
  ConfMaxArgSort(detection, &idx);

  std::vector<float> area;
  CalcArea(detection, &area);

  std::vector<int> pick_idx;
  while (idx.size() > 0) {
    int i = idx.back();
    pick_idx.push_back(i);
    idx.pop_back();

    const cv::Rect_<Dtype>& box1 = detection[i].bbox;

    cv::Rect_<Dtype> avg_box(box1);
    int overlap_cnt = 1;

    auto idx_iter = idx.begin();
    while (idx_iter != idx.end()) {
      const cv::Rect_<Dtype>& box2 = detection[*idx_iter].bbox;
      Dtype x_min = std::max(box1.x, box2.x);
      Dtype y_min = std::max(box1.y, box2.y);
      Dtype x_max = std::min(box1.x + box1.width, box2.x + box2.width);
      Dtype y_max = std::min(box1.y + box1.height, box2.y + box2.height);
      
      Dtype bound_area = 
          std::max(x_max - x_min, static_cast<Dtype>(0)) * 
            std::max(y_max - y_min, static_cast<Dtype>(0));
      float overlap = bound_area / area[*idx_iter];

      if (overlap > overlap_threshold()) {
        idx.erase(idx_iter++);

        avg_box.x += box2.x;
        avg_box.y += box2.y;
        avg_box.width += box2.width;
        avg_box.height += box2.height;
        ++overlap_cnt;
      }
      else
        ++idx_iter;
    }

    avg_box.x /= overlap_cnt;
    avg_box.y /= overlap_cnt;
    avg_box.width /= overlap_cnt;
    avg_box.height /= overlap_cnt;

    Detection<Dtype> mean_detection = detection[i];
    mean_detection.bbox = avg_box;
    result->push_back(mean_detection);
  }

  //result->resize(pick_idx.size());
  //for (int i = 0; i < pick_idx.size(); ++i)
  //  (*result)[i] = detection[pick_idx[i]];
}

template <typename Dtype>
class DistanceNMS : public DetectionNMS<Dtype>
{
  public:
    DistanceNMS(DetectionNMS<Dtype>* base_nms);
    virtual void operator()(const std::vector<Detection<Dtype> >& detection,
                            std::vector<Detection<Dtype> >* result) override;

  private:
    bool IsNear(const Detection<Dtype>& d1,
                const Detection<Dtype>& d2) const;
    float Distance(const Detection<Dtype>& d1,
                   const Detection<Dtype>& d2) const;
    std::unique_ptr<DetectionNMS<Dtype> > base_nms_;
};

template <typename Dtype>
inline DistanceNMS<Dtype>::DistanceNMS(DetectionNMS<Dtype>* base_nms) 
  : base_nms_(base_nms) {

}

template <typename Dtype>
void DistanceNMS<Dtype>::operator()(
    const std::vector<Detection<Dtype> >& detection,
    std::vector<Detection<Dtype> >* result) {
  result->clear();

  std::vector<Detection<Dtype> > base_result;
  (*base_nms_)(detection, &base_result);

  std::list<Detection<Dtype> > temp_result(base_result.begin(), 
                                          base_result.end());
  auto iter1 = temp_result.begin();
  while (iter1 != temp_result.end()) {
    result->push_back(*iter1);

    auto iter2 = iter1;
    ++iter2;
    while (iter2 != temp_result.end()) {
      if (IsNear(*iter1, *iter2))
        temp_result.erase(iter2++);
      else
        ++iter2;
    }

    ++iter1;
  }

  result->assign(temp_result.begin(), temp_result.end());
}

template <typename Dtype>
inline bool DistanceNMS<Dtype>::IsNear(const Detection<Dtype>& d1,
                                       const Detection<Dtype>& d2) const {
  float dist = Distance(d1, d2);
  return (dist < d1.bbox.height * NEAR_DIST);
}

template <typename Dtype>
inline float DistanceNMS<Dtype>::Distance(const Detection<Dtype>& d1,
                                          const Detection<Dtype>& d2) const {
  return std::sqrt(std::pow(d1.bbox.x - d2.bbox.x, 2) + std::pow(d1.bbox.y - d2.bbox.y, 2));
}


} // namespace bgm
#endif // !BGM_DETECTION_NMS_HPP_
