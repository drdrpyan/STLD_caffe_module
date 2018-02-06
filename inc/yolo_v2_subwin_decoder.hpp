#ifndef BGM_YOLO_V2_SUBWIN_DECODER_HPP_
#define BGM_YOLO_V2_SUBWIN_DECODER_HPP_

#include "yolo_v2_decoder.hpp"

namespace bgm
{

template <typename Dtype>
class YOLOV2SubwinDecoder : public YOLOV2Decoder<Dtype>
{
 public:
  YOLOV2SubwinDecoder(YOLOV2Handler<Dtype>* yolo_v2_handler,
                      DetectionNMS<Dtype>* nms,
                      const std::vector<cv::Point>& subwin_offset,
                      bool global_detection);
  virtual void Decode(
      const caffe::Blob<Dtype>& yolo_out, 
      Dtype conf_threshold, bool bbox_nms,
      std::vector<std::vector<Detection<Dtype> > >* detection) override;

 private:
  std::vector<cv::Point> subwin_offset_;
  bool global_detection_;
};

// template functions
template <typename Dtype>
inline YOLOV2SubwinDecoder<Dtype>::YOLOV2SubwinDecoder(
    YOLOV2Handler<Dtype>* yolo_v2_handler,
    DetectionNMS<Dtype>* nms,
    const std::vector<cv::Point>& subwin_offset,
    bool global_detection) 
  : YOLOV2Decoder<Dtype>(yolo_v2_handler, nms), 
    subwin_offset_(subwin_offset), global_detection_(global_detection) {

}

template <typename Dtype>
void YOLOV2SubwinDecoder<Dtype>::Decode(
    const caffe::Blob<Dtype>& yolo_out,
    Dtype conf_threshold, bool bbox_nms,
    std::vector<std::vector<Detection<Dtype> > >* detection) {
  CHECK_EQ(yolo_out.num(), subwin_offset_.size());
  detection->clear();

  std::vector<std::vector<Detection<Dtype> > > temp_detection;
  YOLOV2Decoder<Dtype>::Decode(yolo_out, conf_threshold, false,
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

#endif // !BGM_YOLO_V2_SUBWIN_DECODER_HPP_