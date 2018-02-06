#ifndef BGM_TRAIN_ROI_GENERATOR_HPP_
#define BGM_TRAIN_ROI_GENERATOR_HPP_

#include "obj_contained.hpp"
#include "uniform_integer_rng.hpp"

#include <opencv2/core.hpp>

#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"

#include <vector>
#include <random>
#include <chrono>

namespace bgm
{

template <typename Dtype>
class TrainROIGenerator
{
 public:
  virtual void Generate(
     const std::vector<std::vector<int> >& src_label,
     const std::vector<std::vector<cv::Rect_<Dtype> > >& src_bbox,
     std::vector<std::vector<cv::Rect_<Dtype> > >* roi,
     std::vector<std::vector<int> >* roi_label,
     std::vector<std::vector<cv::Rect_<Dtype> > >* roi_bbox) = 0;

 private:

};

template <typename Dtype>
class TrainROINearGridGenerator : public TrainROIGenerator<Dtype>
{
  struct ROI
  {
    cv::Rect_<Dtype> roi;
    int label;
    cv::Rect_<Dtype> bbox;
  };

 public:
  TrainROINearGridGenerator(int win_width, int win_height,
                            int rows, int cols,
                            Dtype pos_threshold = 0.7,
                            Dtype neg_threshold = 0.1);
  virtual void Generate(
      const std::vector<std::vector<int> >& src_label,
      const std::vector<std::vector<cv::Rect_<Dtype> > >& src_bbox,
      std::vector<std::vector<cv::Rect_<Dtype> > >* roi,
      std::vector<std::vector<int> >* roi_label,
      std::vector<std::vector<cv::Rect_<Dtype> > >* roi_bbox) override;

 private:
  void GenerateNearROI(int label, const cv::Rect_<Dtype>& src_bbox,
                       std::vector<ROI>* near_roi) const;
  void GenerateNegROI(int max_num,
                      const std::vector<int>& src_label,
                      const std::vector<cv::Rect_<Dtype> >& src_bbox,
                      std::vector<ROI>* neg_roi) const;
  cv::Rect_<Dtype> GetCell(const cv::Rect_<Dtype>& src_bbox) const;
  cv::Rect_<Dtype> NormalizeBBox(const cv::Rect_<Dtype>& bbox,
                                 const cv::Rect_<Dtype>& roi) const;
  void ROIToOutput(const std::vector<ROI>& roi,
                   std::vector <cv::Rect_<Dtype> >* out_roi,
                   std::vector<int>* out_label,
                   std::vector <cv::Rect_<Dtype> >* out_bbox) const;
  

  cv::Size win_size_;
  cv::Size cell_size_;
  IntersectionOverObjContained<Dtype> roi_score_;
  Dtype pos_threshold_;
  Dtype neg_threshold_;
  std::shared_ptr<bgm::UniformIntegerRNG<int> > rng_;
};

// template functions
template <typename Dtype>
TrainROINearGridGenerator<Dtype>::TrainROINearGridGenerator(
    int win_width, int win_height, int rows, int cols,
    Dtype pos_threshold, Dtype neg_threshold) 
  : win_size_(win_width, win_height),
    cell_size_(win_width/ cols, win_height / rows),
    pos_threshold_(pos_threshold), neg_threshold_(neg_threshold) {
  CHECK_GT(win_width, 0);
  CHECK_GT(win_height, 0);
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);

  rng_ = bgm::UniformIntegerRNG<int>::GetInstance();
}

template <typename Dtype>
void TrainROINearGridGenerator<Dtype>::Generate(
    const std::vector<std::vector<int> >& src_label,
    const std::vector<std::vector<cv::Rect_<Dtype> > >& src_bbox,
    std::vector<std::vector<cv::Rect_<Dtype> > >* roi,
    std::vector<std::vector<int> >* roi_label,
    std::vector<std::vector<cv::Rect_<Dtype> > >* roi_bbox) {
  CHECK_EQ(src_label.size(), src_bbox.size());
  CHECK(roi);
  CHECK(roi_label);
  CHECK(roi_bbox);
  const int BATCH_SIZE = src_label.size();
  roi->resize(BATCH_SIZE);
  roi_label->resize(BATCH_SIZE);
  roi_bbox->resize(BATCH_SIZE);

  for (int n = 0; n < BATCH_SIZE; ++n) {
    const std::vector<int>& current_src_label = src_label[n];
    const std::vector<cv::Rect_<Dtype> >& current_src_bbox = src_bbox[n];
    CHECK_EQ(current_src_label.size(), current_src_bbox.size());

    std::vector<cv::Rect_<Dtype> >& current_roi = (*roi)[n];
    std::vector<int>& current_roi_label = (*roi_label)[n];
    std::vector<cv::Rect_<Dtype> >& current_roi_bbox = (*roi_bbox)[n];
    current_roi.clear();
    current_roi_label.clear();
    current_roi_bbox.clear();

    std::vector<ROI> generated_roi;
    for (int i = 0; i < current_src_label.size(); ++i) {
      std::vector<ROI> temp_roi;
      GenerateNearROI(current_src_label[i], current_src_bbox[i],
                      &temp_roi);
      generated_roi.insert(generated_roi.end(), 
                           temp_roi.cbegin(), temp_roi.cend());
    }
    std::vector<ROI> temp_roi;
    GenerateNegROI(generated_roi.size() ? 2 : 10, 
                      current_src_label, current_src_bbox,
                      &temp_roi);
    generated_roi.insert(generated_roi.end(), 
                           temp_roi.cbegin(), temp_roi.cend());

    ROIToOutput(generated_roi,
                &current_roi, &current_roi_label, &current_roi_bbox);  
  }
}

template <typename Dtype>
void TrainROINearGridGenerator<Dtype>::GenerateNearROI(
    int label, const cv::Rect_<Dtype>& src_bbox,
    std::vector<ROI>* near_roi) const {
  CHECK(near_roi);
  near_roi->clear();

  cv::Rect_<Dtype> cell = GetCell(src_bbox);
  Dtype half_width = cell_size_.width / 2.;
  Dtype half_height = cell_size_.height / 2.;
  cv::Rect_<Dtype> window(0, 0, win_size_.width, win_size_.height);

  std::vector<cv::Rect_<Dtype> > candidate;

  candidate.push_back(cv::Rect_<Dtype>(cell.x - half_width,
                                       cell.y - half_height,
                                       cell.width, cell.height));
  candidate.push_back(cv::Rect_<Dtype>(cell.x,
                                       cell.y - half_height,
                                       cell.width, cell.height));
  candidate.push_back(cv::Rect_<Dtype>(cell.x + half_width,
                                       cell.y - half_height,
                                       cell.width, cell.height));
  
  candidate.push_back(cv::Rect_<Dtype>(cell.x - half_width,
                                       cell.y,
                                       cell.width, cell.height));
  candidate.push_back(cv::Rect_<Dtype>(cell.x,
                                       cell.y,
                                       cell.width, cell.height));
  candidate.push_back(cv::Rect_<Dtype>(cell.x + half_width,
                                       cell.y,
                                       cell.width, cell.height));

  candidate.push_back(cv::Rect_<Dtype>(cell.x - half_width,
                                       cell.y + half_height,
                                       cell.width, cell.height));
  candidate.push_back(cv::Rect_<Dtype>(cell.x,
                                       cell.y + half_height,
                                       cell.width, cell.height));
  candidate.push_back(cv::Rect_<Dtype>(cell.x + half_width,
                                       cell.y + half_height,
                                       cell.width, cell.height));

  candidate.push_back(cv::Rect_<Dtype>(cell.x - cell_size_.width,
                                       cell.y - cell_size_.height,
                                       cell.width * 2, cell.height * 2));

  for (auto iter = candidate.cbegin(); iter != candidate.cend(); ++iter) {
    Dtype roi_score = roi_score_.Score(*iter, src_bbox);
    if (roi_score > pos_threshold_) {
      if (iter->height == cell.height * 2) {
        if(src_bbox.height > 30)
          near_roi->push_back({*iter, label, NormalizeBBox(src_bbox, *iter)});
      }
      else
        near_roi->push_back({*iter, label, NormalizeBBox(src_bbox, *iter)});
    }
    //else if (roi_score < neg_threshold_) {
    //  near_roi->push_back({*iter, caffe::LabelParameter::NONE,
    //                       cv::Rect_<Dtype>(caffe::BBoxParameter::DUMMY_VALUE,
    //                                        caffe::BBoxParameter::DUMMY_VALUE,
    //                                        caffe::BBoxParameter::DUMMY_VALUE,
    //                                        caffe::BBoxParameter::DUMMY_VALUE)});
    //}
  }
}

template <typename Dtype>
void TrainROINearGridGenerator<Dtype>::GenerateNegROI(
    int max_num, const std::vector<int>& src_label,
    const std::vector<cv::Rect_<Dtype> >& src_bbox,
    std::vector<ROI>* neg_roi) const {
  CHECK_GE(max_num, 0);
  CHECK_EQ(src_label.size(), src_bbox.size());
  CHECK(neg_roi);

  std::vector<cv::Rect_<Dtype> > candidate;

  int single_x_max = win_size_.width - cell_size_.width;
  int single_y_max = win_size_.height - cell_size_.height;
  for (int i = max_num / 2; i--; ) {
    int x = rng_->Random(0, single_x_max);
    int y = rng_->Random(0, single_y_max);
    candidate.push_back(cv::Rect_<Dtype>(x, y,
                                         cell_size_.width,
                                         cell_size_.height));
  }

  int double_x_max = win_size_.width - (cell_size_.width * 2);
  int double_y_max = win_size_.height - (cell_size_.height * 2);
  for (int i = max_num / 2; i--; ) {
    int x = rng_->Random(0, double_x_max);
    int y = rng_->Random(0, double_y_max);
    candidate.push_back(cv::Rect_<Dtype>(x, y,
                                         cell_size_.width * 2,
                                         cell_size_.height * 2));
  }

  for (auto roi = candidate.cbegin(); roi != candidate.cend(); ++roi) {
    Dtype max_roi_score = 0;
    for (auto bbox = src_bbox.cbegin(); bbox != src_bbox.cend(); ++bbox) {
      Dtype roi_score = roi_score_.Score(*roi, *bbox);
      if (roi_score > max_roi_score)
        max_roi_score = roi_score;
    }

    if (max_roi_score < neg_threshold_)
      neg_roi->push_back({*roi, caffe::LabelParameter::NONE,
                          cv::Rect_<Dtype>(caffe::BBoxParameter::DUMMY_VALUE,
                                           caffe::BBoxParameter::DUMMY_VALUE,
                                           caffe::BBoxParameter::DUMMY_VALUE,
                                           caffe::BBoxParameter::DUMMY_VALUE)});
  }  
}

template <typename Dtype>
inline cv::Rect_<Dtype> TrainROINearGridGenerator<Dtype>::GetCell(
    const cv::Rect_<Dtype>& src_bbox) const {
  Dtype x = std::floor(src_bbox.x / cell_size_.width) * cell_size_.width;
  Dtype y = std::floor(src_bbox.y / cell_size_.height) * cell_size_.height;
  return cv::Rect_<Dtype>(x, y, cell_size_.width, cell_size_.height);
}

template <typename Dtype>
cv::Rect_<Dtype> TrainROINearGridGenerator<Dtype>::NormalizeBBox(
    const cv::Rect_<Dtype>& bbox, const cv::Rect_<Dtype>& roi) const {
  Dtype center_x = bbox.x + (bbox.width / 2.) - roi.x;
  Dtype center_y = bbox.y + (bbox.height / 2.) - roi.y;
  center_x /= roi.width;
  center_y /= roi.height;

  Dtype log_w = std::log(bbox.width / roi.width);
  Dtype log_h = std::log(bbox.height / roi.height);

  return cv::Rect_<Dtype>(center_x, center_y, log_w, log_h);
}

template <typename Dtype>
void TrainROINearGridGenerator<Dtype>::ROIToOutput(
    const std::vector<ROI>& roi,
    std::vector <cv::Rect_<Dtype> >* out_roi,
    std::vector<int>* out_label,
    std::vector <cv::Rect_<Dtype> >* out_bbox) const {
  CHECK(out_roi);
  CHECK(out_label);
  CHECK(out_bbox);
  const int NUM_ROI = roi.size();
  out_roi->resize(NUM_ROI);
  out_label->resize(NUM_ROI);
  out_bbox->resize(NUM_ROI);

  for (int i = 0; i < NUM_ROI; ++i) {
    (*out_roi)[i] = roi[i].roi;
    (*out_label)[i] = roi[i].label;
    (*out_bbox)[i] = roi[i].bbox;
  }
}
} // namespace bgm
#endif // !BGM_TRAIN_ROI_GENERATOR_HPP_
