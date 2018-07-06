#ifndef BGM_DHA_SUBWIN_DECODER_HPP_
#define BGM_DHA_SUBWIN_DECODER_HPP_

#include "dha_decoder.hpp"

namespace bgm
{

template <typename Dtype>
class DHASubwinDecoder : public DHADecoder<Dtype>
{
 public:
  DHASubwinDecoder(DHAHandler<Dtype>* dha_handler,
                   DetectionNMS<Dtype>* nms,
                   const std::vector<float>& anchor_weight = std::vector<float>(),
                   const std::vector<float>& class_weight = std::vector<float>(),
                   bool class_label_zero_begin = false,
                   const std::vector<cv::Point>& subwin_offset_ = std::vector<cv::Point>(),
                   bool global_detection = false);
  virtual void Decode(
      const caffe::Blob<Dtype>& dha_out, 
      Dtype neg_threshold, bool bbox_nms,
      std::vector<std::vector<Detection<Dtype> > >* detection) override;

 private:
  std::vector<cv::Point> subwin_offset_;
  bool global_detection_;

}; // class DHASubwinDecoder

// template fucntions
template <typename Dtype>
DHASubwinDecoder<Dtype>::DHASubwinDecoder(
    DHAHandler<Dtype>* dha_handler, DetectionNMS<Dtype>* nms,
    const std::vector<float>& anchor_weight,
    const std::vector<float>& class_weight,
    bool class_label_zero_begin,
    const std::vector<cv::Point>& subwin_offset,
    bool global_detection) 
  : DHADecoder<Dtype>(dha_handler, nms, 
                      anchor_weight, class_weight, 
                      class_label_zero_begin),
    subwin_offset_(subwin_offset), global_detection_(global_detection) {

}

template <typename Dtype>
void DHASubwinDecoder<Dtype>::Decode(
    const caffe::Blob<Dtype>& dha_out,
    Dtype neg_threshold, bool bbox_nms,
    std::vector<std::vector<Detection<Dtype> > >* detection) {
  CHECK_EQ(dha_out.num(), subwin_offset_.size());
  CHECK(detection);
  detection->clear();

  std::vector<std::vector<Detection<Dtype> > > temp_detection;
  DHADecoder<Dtype>::Decode(dha_out, neg_threshold, false,
                            &temp_detection);

  for (int i = 0; i < temp_detection.size(); ++i) {
    const cv::Point& current_offset = subwin_offset_[i];
    std::vector<Detection<Dtype> >& current_detection = temp_detection[i];

    for (auto iter = current_detection.begin(); iter != current_detection.end();
         ++iter) {
      iter->bbox.x += current_offset.x;
      iter->bbox.y += current_offset.y;
    }
  }

  if (global_detection_) {
    detection->resize(1);

    for (int i = 1; i < temp_detection.size(); ++i) {
      temp_detection[0].insert(temp_detection[0].end(),
                               temp_detection[i].cbegin(),
                               temp_detection[i].cend());
      //temp_detection[i].clear();
    }
    if (bbox_nms)
      nms()(temp_detection[0], &((*detection)[0]));
    else
      (*detection)[0].assign(temp_detection[0].cbegin(),
                             temp_detection[0].cend());
  }
  else {
    detection->resize(temp_detection.size());

    for (int i = 0; i < temp_detection.size(); ++i) {
      if (bbox_nms)
        nms()(temp_detection[i], &((*detection)[i]));
      else
        (*detection)[i].assign(temp_detection[i].cbegin(),
                               temp_detection[i].cend());
    }
  }
}

} // namespace bgm

#endif // !BGM_DHA_SUBWIN_DECODER_HPP_
