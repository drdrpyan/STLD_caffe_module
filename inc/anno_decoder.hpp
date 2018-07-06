#ifndef BGM_ANNO_DECODER_HPP_
#define BGM_ANNO_DECODER_HPP_

#include "detection_util.hpp"

#include <opencv2/core.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"

#include <set>
#include <vector>
#include <iterator>

namespace bgm
{

template <typename Dtype>
class AnnoDecoder
{
 public:
  AnnoDecoder();
  virtual void Decode(
      const std::vector<caffe::Blob<Dtype>*>& blobs,
      std::vector<std::vector<bgm::BoxAnnotation<Dtype> > >* detection);
  virtual void Decode(const std::vector<caffe::Blob<Dtype>*>& blobs,
                      std::vector<std::vector<int> >* label,
                      std::vector<std::vector<cv::Rect_<Dtype> > >* bbox);

  void AddIgnoreLabel(int label);
  template <typename Iterator>
  void AddIgnoreLabel(const Iterator& first, const Iterator& end);

 protected:
  bool IsIgnoreLabel(int label) const;
  std::set<int> ignore_label_;
};

// template functions
template <typename Dtype>
inline AnnoDecoder<Dtype>::AnnoDecoder() {
  AddIgnoreLabel(caffe::LabelParameter::NONE);
  AddIgnoreLabel(caffe::LabelParameter::DUMMY_LABEL);
}

template <typename Dtype>
void AnnoDecoder<Dtype>::Decode(
    const std::vector<caffe::Blob<Dtype>*>& blobs,
    std::vector<std::vector<bgm::BoxAnnotation<Dtype> > >* detection) {
  CHECK(detection);

  std::vector<std::vector<int> > label;
  std::vector<std::vector<cv::Rect_<Dtype> > > bbox;
  Decode(blobs, &label, &bbox);
  CHECK_EQ(label.size(), bbox.size());

  detection->resize(label.size());
  for (int i = 0; i < detection->size(); ++i) {
    CHECK_EQ(label[i].size(), bbox[i].size());
    (*detection)[i].resize(label[i].size());

    for (int j = 0; j < label[i].size(); ++j) {
      (*detection)[i][j].label = label[i][j];
      (*detection)[i][j].bbox = bbox[i][j];
    }
  }
}

template <typename Dtype>
void AnnoDecoder<Dtype>::Decode(
    const std::vector<caffe::Blob<Dtype>*>& blobs,
    std::vector<std::vector<int> >* label,
    std::vector<std::vector<cv::Rect_<Dtype> > >* bbox) {
  CHECK_EQ(blobs.size(), 2);
  CHECK(label);
  CHECK(bbox);

  const caffe::Blob<Dtype>& label_blob = *(blobs[0]);
  const caffe::Blob<Dtype>& bbox_blob = *(blobs[1]);
  CHECK_EQ(label_blob.num(), bbox_blob.num());
  CHECK_EQ(label_blob.channels(), 1);
  CHECK_EQ(bbox_blob.channels(), 4);
  CHECK_EQ(label_blob.count(2), bbox_blob.count(2));

  const Dtype* label_ptr = label_blob.cpu_data();
  const Dtype* bbox_ptr = bbox_blob.cpu_data();
  
  const int BATCH_SIZE = label_blob.num();
  label->resize(BATCH_SIZE);
  bbox->resize(BATCH_SIZE);

  const int MAX_ANNO = label_blob.count(2);

  for (int n = 0; n < BATCH_SIZE; ++n) {
    std::vector<int>& temp_label = (*label)[n];
    std::vector<cv::Rect_<Dtype> >& temp_bbox = (*bbox)[n];
    temp_label.clear();
    temp_bbox.clear();

    const Dtype* label_iter = label_ptr + label_blob.offset(n);
    const Dtype* x_iter = bbox_ptr + bbox_blob.offset(n, 0);
    const Dtype* y_iter = bbox_ptr + bbox_blob.offset(n, 1);
    const Dtype* w_iter = bbox_ptr + bbox_blob.offset(n, 2);
    const Dtype* h_iter = bbox_ptr + bbox_blob.offset(n, 3);

    for (int i = MAX_ANNO; i--; ) {
      if (!IsIgnoreLabel(*label_iter)) {
        temp_label.push_back(*label_iter);
        temp_bbox.push_back(cv::Rect_<Dtype>(*x_iter, *y_iter,
                                             *w_iter, *h_iter));
      }

      ++label_iter;
      ++x_iter;
      ++y_iter;
      ++w_iter;
      ++h_iter;
    }
  }
}

template <typename Dtype>
void AnnoDecoder<Dtype>::AddIgnoreLabel(int label) {
  ignore_label_.insert(label);
}

template <typename Dtype>
template <typename  Iterator>
void AnnoDecoder<Dtype>::AddIgnoreLabel(
    const Iterator& first, const Iterator& end) {
  Iterator iter = first;
  while (iter != end) {
    ignore_label_.insert(*iter);
    ++iter;
  }
}

template <typename Dtype>
inline bool AnnoDecoder<Dtype>::IsIgnoreLabel(int label) const {
  auto iter = ignore_label_.find(label);
  return (iter != ignore_label_.cend());
}
} // namespace bgm

#endif // !BGM_ANNO_DECODER_HPP_
