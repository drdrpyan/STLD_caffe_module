#ifndef BGM_ANNO_ENCODER_HPP_
#define BGM_ANNO_ENCODER_HPP_

#include <opencv2/core.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"

#include <vector>

namespace bgm
{

template <typename Dtype>
class AnnoEncoder
{
 public:
  //AnnoEncoder();
  virtual void Encode(
      const std::vector<std::vector<int> >& label,
      const std::vector<std::vector<cv::Rect_<Dtype> > >& bbox,
      std::vector<caffe::Blob<Dtype>*>& blobs);

}; // class AnnoEncoder

// template functions
//template <typename Dtype>
//inline AnnoEncoder<Dtype>::AnnoEncoder() { }

template <typename Dtype>
void AnnoEncoder<Dtype>::Encode(
    const std::vector<std::vector<int> >& label,
    const std::vector<std::vector<cv::Rect_<Dtype> > >& bbox,
    std::vector<caffe::Blob<Dtype>*>& blobs) {
  CHECK_EQ(label.size(), bbox.size());
  CHECK_EQ(blobs.size(), 2);
  caffe::Blob<Dtype>& label_blob = *(blobs[0]);
  caffe::Blob<Dtype>& bbox_blob = *(blobs[1]);
  CHECK_EQ(label.size(), label_blob.num());
  CHECK_EQ(label_blob.num(), bbox_blob.num());
  CHECK_EQ(label_blob.channels(), 1);
  CHECK_EQ(bbox_blob.channels(), 4);
  CHECK_EQ(label_blob.height(), bbox_blob.height());

  Dtype* label_ptr = label_blob.mutable_cpu_data();
  Dtype* bbox_ptr = bbox_blob.mutable_cpu_data();

  caffe::caffe_set(label_blob.count(),
                   static_cast<Dtype>(caffe::LabelParameter::NONE),
                   label_ptr);
  caffe::caffe_set(bbox_blob.count(),
                   static_cast<Dtype>(caffe::BBoxParameter::DUMMY_VALUE),
                   bbox_ptr);

  for (int n = 0; n < label.size(); ++n) {
    CHECK_EQ(label[n].size(), bbox[n].size());
    CHECK_LE(label[n].size(), label_blob.height());

    Dtype* label_iter = label_ptr + label_blob.offset(n);
    Dtype* x_iter = bbox_ptr + bbox_blob.offset(n, 0);
    Dtype* y_iter = bbox_ptr + bbox_blob.offset(n, 1);
    Dtype* w_iter = bbox_ptr + bbox_blob.offset(n, 2);
    Dtype* h_iter = bbox_ptr + bbox_blob.offset(n, 3);

    for (int i = 0; i < label[n].size(); ++i) {
      int l = label[n][i];
      const cv::Rect_<Dtype>& b = bbox[n][i];
      if (l != caffe::LabelParameter::NONE && l != caffe::LabelParameter::DUMMY_LABEL) {
        *label_iter = l;
        *x_iter = b.x;
        *y_iter = b.y;
        *w_iter = b.width;
        *h_iter = b.height;
      }
      else {
        *label_iter = caffe::LabelParameter::NONE;
        *x_iter = caffe::BBoxParameter::DUMMY_VALUE;
        *y_iter = caffe::BBoxParameter::DUMMY_VALUE;
        *w_iter = caffe::BBoxParameter::DUMMY_VALUE;
        *h_iter = caffe::BBoxParameter::DUMMY_VALUE;
      }

      ++label_iter;
      ++x_iter;
      ++y_iter;
      ++w_iter;
      ++h_iter;
    }
  }
}

} // namespace bgm

#endif // !BGM_ANNO_ENCODER_HPP_
