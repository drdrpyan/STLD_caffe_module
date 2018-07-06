#ifndef BGM_DETECTION_DECODER_HPP_
#define BGM_DETECTION_DECODER_HPP_

#include "detection_util.hpp"
#include "anno_decoder.hpp"

#include "caffe/blob.hpp"

namespace bgm
{

//template <typename Dtype>
//class DetectionDecoder
//{
// public:
//  DetectionDecoder();
//  virtual void Decode(
//      const std::vector<caffe::Blob<Dtype>*>& blobs,
//      std::vector<std::vector<Detection<Dtype> > >* detection);
//
// private:
//  void DecodeConf(const caffe::Blob<Dtype>& conf_blob,
//                  const std::vector<std::vector<int> >& label,
//                  std::vector<std::vector<Dtype> >* conf);
//  std::unique_ptr<AnnoDecoder<Dtype> > anno_decoder_;
//}; // class DetectionDecoder
//
//// template functions
//template <typename Dtype>
//inline DetectionDecoder<Dtype>::DetectionDecoder() 
//  : anno_decoder_(new AnnoDecoder<Dtype>) {
//
//}
//
//template <typename Dtype>
//void DetectionDecoder<Dtype>::Decode(
//    const std::vector<caffe::Blob<Dtype>*>& blobs,
//    std::vector<std::vector<Detection<Dtype> > >* detection) {
//  CHECK(detection);
//
//  std::vector<caffe::Blob<Dtype>*> label_bbox(2);
//  label_bbox[0] = blobs[0];
//  label_bbox[1] = blobs[1];
//  std::vector<std::vector<int> > label;
//  std::vector<std::vector<cv::Rect_<Dtype> > > bbox;
//  anno_decoder_->Decode(label_bbox, &label, &bbox);
//
//  std::vector<std::vector<Dtype> > conf;
//  DecodeConf(*(blobs[2]), label, &conf);
//
//  CHECK_EQ(label.size(), bbox.size());
//  CHECK_EQ(bbox.size(), conf.size());
//
//  detection->resize(label.size());
//  for (int n = 0; n < detection->size(); ++n) {
//    std::vector<Detection<Dtype> >& batch_detection = (*detection)[n];
//    const std::vector<int>& batch_label = label[n];
//    const std::vector<cv::Rect_<Dtype> >& batch_bbox = bbox[n];
//    const std::vector<Dtype>& batch_conf = conf[n];
//
//    CHECK_EQ(batch_label.size(), batch_bbox.size());
//    CHECK_EQ(batch_bbox.size(), batch_conf.size());
//
//    batch_detection.resize(batch_label.size());
//    for (int i = 0; i < batch_label.size(); ++i) {
//      batch_detection[i].label = batch_label[i];
//      batch_detection[i].bbox = batch_bbox[i];
//      batch_detection[i].conf = batch_conf[i];
//    }
//  }
//}
//
//template <typename Dtype>
//void DetectionDecoder<Dtype>::DecodeConf(
//    const caffe::Blob<Dtype>& conf_blob,
//    const std::vector<std::vector<int> >& label,
//    std::vector<std::vector<Dtype> >* conf) {
//  CHECK_EQ(conf_blob.channels(), 1);
//  CHECK(conf);
//
//  conf->resize(conf_blob.num());
//
//  const Dtype* conf_ptr = conf_blob.cpu_data();
//  for (int n = 0; n < conf->size(); ++n) {
//    std::vector<Dtype>& batch_conf = (*conf)[n];
//    batch_conf.resize(label[n].size());
//
//    const Dtype* conf_iter = conf_ptr + conf_blob.offset(n);
//    for (int i = 0; i < batch_conf.size(); ++i)
//      batch_conf[i] = *conf_iter++;
//  }
//}

template <typename Dtype>
class DetectionDecoder : public AnnoDecoder<Dtype>
{
 public:
  virtual void Decode(
      const std::vector<caffe::Blob<Dtype>*>& blobs,
      std::vector<std::vector<Detection<Dtype> > >* detection);

 private:
}; // class DetectionDecoder

// template functions
template <typename Dtype>
void DetectionDecoder<Dtype>::Decode(
    const std::vector<caffe::Blob<Dtype>*>& blobs,
    std::vector<std::vector<Detection<Dtype> > >* detection) {
  CHECK_EQ(blobs.size(), 3);
  CHECK(detection);

  const caffe::Blob<Dtype>& label_blob = *(blobs[0]);
  const caffe::Blob<Dtype>& bbox_blob = *(blobs[1]);
  const caffe::Blob<Dtype>& conf_blob = *(blobs[2]);
  CHECK_EQ(label_blob.num(), bbox_blob.num());
  CHECK_EQ(label_blob.channels(), 1);
  CHECK_EQ(bbox_blob.channels(), 4);
  CHECK_EQ(label_blob.count(2), bbox_blob.count(2));
  CHECK(label_blob.shape() == conf_blob.shape());

  const Dtype* label_ptr = label_blob.cpu_data();
  const Dtype* bbox_ptr = bbox_blob.cpu_data();
  const Dtype* conf_ptr = conf_blob.cpu_data();
  
  const int BATCH_SIZE = label_blob.num();
  detection->resize(BATCH_SIZE);  

  const int MAX_ANNO = label_blob.count(2);

  for (int n = 0; n < BATCH_SIZE; ++n) {
    (*detection)[n].clear();

    const Dtype* label_iter = label_ptr + label_blob.offset(n);
    const Dtype* x_iter = bbox_ptr + bbox_blob.offset(n, 0);
    const Dtype* y_iter = bbox_ptr + bbox_blob.offset(n, 1);
    const Dtype* w_iter = bbox_ptr + bbox_blob.offset(n, 2);
    const Dtype* h_iter = bbox_ptr + bbox_blob.offset(n, 3);
    const Dtype* conf_iter = conf_ptr + conf_blob.offset(n);

    for (int i = MAX_ANNO; i--; ) {
      if (!IsIgnoreLabel(*label_iter)) {
        Detection<Dtype> temp_detection;
        temp_detection.label = *label_iter;
        temp_detection.bbox = cv::Rect_<Dtype>(*x_iter, *y_iter,
                                               *w_iter, *h_iter);
        temp_detection.conf = *conf_iter;

        (*detection)[n].push_back(temp_detection);
      }

      ++label_iter;
      ++x_iter;
      ++y_iter;
      ++w_iter;
      ++h_iter;
      ++conf_iter;
    }
  }
}
} // namespace bgm

#endif // !BGM_DETECTION_DECODER_HPP_
