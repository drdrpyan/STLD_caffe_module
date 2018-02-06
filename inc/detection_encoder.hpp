#ifndef BGM_DETECTION_ENCODER_HPP_
#define BGM_DETECTION_ENCODER_HPP_

#include "detection_util.hpp"

#include "anno_encoder.hpp"

#include "caffe/blob.hpp"

namespace bgm
{

template <typename Dtype>
class DetectionEncoder
{
 public:
  DetectionEncoder();
  virtual void Encode(
      const std::vector<std::vector<Detection<Dtype> > >& detection,
      const std::vector<caffe::Blob<Dtype>*>& blobs);

 private:
  void EncodeLabelBBox(
      const std::vector<std::vector<Detection<Dtype> > >& detection,
      const std::vector<caffe::Blob<Dtype>*>& blobs);

  std::unique_ptr<bgm::AnnoEncoder<Dtype> > anno_encoder_;
}; // class DetectionEncoder

// template functions
template <typename Dtype>
inline DetectionEncoder<Dtype>::DetectionEncoder()
  : anno_encoder_(new bgm::AnnoEncoder<Dtype>){
}

template <typename Dtype>
void DetectionEncoder<Dtype>::Encode(
    const std::vector<std::vector<Detection<Dtype> > >& detection,
    const std::vector<caffe::Blob<Dtype>*>& blobs) {
  CHECK_EQ(blobs.size(), 3);

  EncodeLabelBBox(detection, blobs);

  caffe::Blob<Dtype>& conf_blob = *(blobs[2]);
  CHECK_EQ(conf_blob.num(), detection.size());
  CHECK_EQ(conf_blob.channels(), 1);

  Dtype* conf_ptr = conf_blob.mutable_cpu_data();
  for (int n = 0; n < detection.size(); ++n) {
    const std::vector<Detection<Dtype> >& batch_detection = detection[n];
    CHECK_LE(batch_detection.size(), conf_blob.height());

    Dtype* conf_iter = conf_ptr + conf_blob.offset(n);
    int i;    
    for (i = 0; i < batch_detection.size(); ++i)
      *conf_iter++ = batch_detection[i].conf;
    for (; i < conf_blob.height(); ++i)
      *conf_iter++ = 0;
  }
}

template <typename Dtype>
void DetectionEncoder<Dtype>::EncodeLabelBBox(
    const std::vector<std::vector<Detection<Dtype> > >& detection,
    const std::vector<caffe::Blob<Dtype>*>& blobs) {
  std::vector<std::vector<int> > label(detection.size());
  std::vector<std::vector<cv::Rect_<Dtype> > > bbox(detection.size());
  for (int n = 0; n < detection.size(); ++n) {
    const std::vector<Detection<Dtype> >& batch_detection = detection[n];
    std::vector<int>& batch_label = label[n];
    std::vector<cv::Rect_<Dtype> >& batch_bbox = bbox[n];

    batch_label.resize(batch_detection.size());
    batch_bbox.resize(batch_detection.size());

    for (int i = 0; i < batch_detection.size(); ++i) {
      batch_label[i] = batch_detection[i].label;
      batch_bbox[i] = batch_detection[i].bbox;
    }
  }

  std::vector<caffe::Blob<Dtype>*> label_bbox(2);
  label_bbox[0] = blobs[0];
  label_bbox[1] = blobs[1];

  anno_encoder_->Encode(label, bbox, label_bbox);
}

} // namespace bgm

#endif // !BGM_DETECTION_ENCODER_HPP_
