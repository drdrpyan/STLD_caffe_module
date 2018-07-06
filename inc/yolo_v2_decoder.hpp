#ifndef BGM_YOLO_V2_DECODER_HPP_
#define BGM_YOLO_V2_DECODER_HPP_

#include "detection_util.hpp"
#include "yolo_v2_handler.hpp"
#include "detection_nms.hpp"

#include "caffe/blob.hpp"

#include <memory>

namespace bgm
{

template <typename Dtype>
class YOLOV2Decoder
{
 public:
  YOLOV2Decoder(YOLOV2Handler<Dtype>* yolo_v2_handler,
                DetectionNMS<Dtype>* nms);
  virtual void Decode(const caffe::Blob<Dtype>& yolo_out, 
                      Dtype conf_threshold, bool bbox_nms,
                      std::vector<std::vector<Detection<Dtype> > >* detection);

 protected:
  DetectionNMS<Dtype>& nms();
  void InitCellData(const caffe::Blob<Dtype>& yolo_out, int batch_idx);
  virtual void DecodeCellData(int h, int w, Dtype conf_threshold, 
                              std::vector<Detection<Dtype> >* detection) const;
  cv::Rect_<Dtype> GetYOLOBox(int anchor) const;
  void GetMaxClass(int anchor_idx,
                   int* max_class_idx, float* max_class_conf) const;

  void NextCellData();

  std::vector<const Dtype*> cell_data_;

  std::unique_ptr<YOLOV2Handler<Dtype> > yolo_v2_handler_;
  std::unique_ptr<DetectionNMS<Dtype> > nms_;
};

// template functions
template <typename Dtype>
YOLOV2Decoder<Dtype>::YOLOV2Decoder(
    YOLOV2Handler<Dtype>* yolo_v2_handler, DetectionNMS<Dtype>* nms) 
  : yolo_v2_handler_(yolo_v2_handler), nms_(nms) {

}

template <typename Dtype>
void YOLOV2Decoder<Dtype>::Decode(
    const caffe::Blob<Dtype>& yolo_out,
    Dtype conf_threshold, bool bbox_nms,
    std::vector<std::vector<Detection<Dtype> > >* detection) {
  CHECK(detection);
  detection->resize(yolo_out.num());

  for (int n = 0; n < yolo_out.num(); ++n) {
    InitCellData(yolo_out, n);
    std::vector<Detection<Dtype> >& batch_detection = (*detection)[n];
    batch_detection.clear();

    for (int h = 0; h < yolo_out.height(); ++h) {
      for (int w = 0; w < yolo_out.width(); ++w) {
        std::vector<Detection<Dtype> > cell_detection;
        DecodeCellData(h, w, conf_threshold, &cell_detection);

        batch_detection.insert(batch_detection.end(),
                               cell_detection.begin(), cell_detection.end());

        NextCellData();
      }
    }

    if (bbox_nms) {
      std::vector<Detection<Dtype> > nms_detection;
      (*nms_)(batch_detection, &nms_detection);
      batch_detection.assign(nms_detection.begin(), nms_detection.end());
    }
  }
}

template <typename Dtype>
inline DetectionNMS<Dtype>& YOLOV2Decoder<Dtype>::nms() {
  return *nms_;
}

template <typename Dtype>
void YOLOV2Decoder<Dtype>::InitCellData(
    const caffe::Blob<Dtype>& yolo_out, int batch_idx) {
  cell_data_.resize(yolo_v2_handler_->NumChannels());
  
  const Dtype* ptr = yolo_out.cpu_data() + yolo_out.offset(batch_idx);
  int step = yolo_out.count(2);
  for (auto iter = cell_data_.begin(); iter != cell_data_.end(); ++iter) {
    *iter = ptr;
    ptr += step;
  }
}

template <typename Dtype>
void YOLOV2Decoder<Dtype>::DecodeCellData(
    int h, int w, Dtype conf_threshold,
    std::vector<Detection<Dtype> >* detection) const {
  CHECK(detection);
  detection->clear();

  for (int i = 0; i < yolo_v2_handler_->anchor().size(); ++i) {
    int conf_ch = yolo_v2_handler_->GetAnchorChannel(i, YOLOV2Handler<Dtype>::AnchorChannel::CONF);
    Dtype conf = bgm::Sigmoid<Dtype>(*(cell_data_[conf_ch]));

    // 기존 conf 취급
    //if (conf > conf_threshold) {
    //  cv::Rect_<Dtype> yolo_box = GetYOLOBox(i);
    //  cv::Rect_<Dtype> raw_box = yolo_v2_handler_->YOLOBoxToRawBox(yolo_box, h, w, i);

    //  int class_idx;
    //  float class_conf;
    //  GetMaxClass(i, &class_idx, &class_conf);

    //  detection->push_back({class_idx + 1, raw_box, static_cast<float>(conf)});
    //}

    // class conf와 조합된 conf
    int class_idx;
    float class_conf;
    GetMaxClass(i, &class_idx, &class_conf);

    conf *= class_conf;

    if (conf > conf_threshold) {
      cv::Rect_<Dtype> yolo_box = GetYOLOBox(i);
      cv::Rect_<Dtype> raw_box = yolo_v2_handler_->YOLOBoxToRawBox(yolo_box, h, w, i);

      detection->push_back({class_idx + 1, raw_box, static_cast<float>(conf)});
    }
  }
}

template <typename Dtype>
cv::Rect_<Dtype> YOLOV2Decoder<Dtype>::GetYOLOBox(int anchor) const {
  Dtype x = *(cell_data_[yolo_v2_handler_->GetAnchorChannel(anchor, YOLOV2Handler<Dtype>::AnchorChannel::X)]);
  Dtype y = *(cell_data_[yolo_v2_handler_->GetAnchorChannel(anchor, YOLOV2Handler<Dtype>::AnchorChannel::Y)]);
  Dtype w = *(cell_data_[yolo_v2_handler_->GetAnchorChannel(anchor, YOLOV2Handler<Dtype>::AnchorChannel::W)]);
  Dtype h = *(cell_data_[yolo_v2_handler_->GetAnchorChannel(anchor, YOLOV2Handler<Dtype>::AnchorChannel::H)]);
  return cv::Rect_<Dtype>(x, y, w, h);
}

template <typename Dtype>
void YOLOV2Decoder<Dtype>::GetMaxClass(int anchor_idx,
                                       int* max_class_idx,
                                       float* max_class_conf) const {
  CHECK(max_class_idx);
  CHECK(max_class_conf);

  *max_class_idx = -1;
  *max_class_conf = -(std::numeric_limits<float>::max());

  int idx = yolo_v2_handler_->GetAnchorChannel(anchor_idx, YOLOV2Handler<Dtype>::AnchorChannel::CLASS_BEGIN);
  for (int i = 0; i < yolo_v2_handler_->num_class(); ++i) {
    Dtype class_conf = *(cell_data_[idx + i]);
    if (class_conf > *max_class_conf) {
      *max_class_idx = i;
      *max_class_conf = class_conf;
    }
  }
}

template <typename Dtype>
inline void YOLOV2Decoder<Dtype>::NextCellData() {
  for (auto iter = cell_data_.begin(); iter != cell_data_.end(); ++iter)
    ++(*iter);
}

template <typename Dtype>
class YOLOV2SoftmaxDecoder : public YOLOV2Decoder<Dtype>
{
 public:
  YOLOV2SoftmaxDecoder(YOLOV2Handler<Dtype>* yolo_v2_handler,
                       DetectionNMS<Dtype>* nms);
  YOLOV2SoftmaxDecoder(YOLOV2Handler<Dtype>* yolo_v2_handler,
                       DetectionNMS<Dtype>* nms,
                       const std::vector<float>& anchor_weight);
 protected:
  virtual void DecodeCellData(
      int h, int w, Dtype conf_threshold, 
      std::vector<Detection<Dtype> >* detection) const override;

 private:
  void ClsSoftmax(int anchor_idx,
                  std::vector<Dtype>* cls_conf, int* max_idx) const;

  std::vector<float> anchor_weight_;

}; // class YOLOV2SoftmaxDecoder

template <typename Dtype>
inline YOLOV2SoftmaxDecoder<Dtype>::YOLOV2SoftmaxDecoder(
    YOLOV2Handler<Dtype>* yolo_v2_handler, DetectionNMS<Dtype>* nms) 
  : YOLOV2Decoder<Dtype>(yolo_v2_handler, nms) {
  anchor_weight_.assign(yolo_v2_handler_->anchor().size(), 1.0f);
}

template <typename Dtype>
inline YOLOV2SoftmaxDecoder<Dtype>::YOLOV2SoftmaxDecoder(
    YOLOV2Handler<Dtype>* yolo_v2_handler,
    DetectionNMS<Dtype>* nms, const std::vector<float>& anchor_weight) 
  : YOLOV2SoftmaxDecoder<Dtype>(yolo_v2_handler, nms) {
  anchor_weight_.assign(anchor_weight.begin(), anchor_weight.end());
  CHECK_EQ(anchor_weight_.size(), yolo_v2_handler_->anchor().size());
}
template <typename Dtype>
void YOLOV2SoftmaxDecoder<Dtype>::DecodeCellData(
    int h, int w, Dtype conf_threshold,
    std::vector<Detection<Dtype> >* detection) const {
  CHECK(detection);
  detection->clear();

  for (int i = 0; i < yolo_v2_handler_->anchor().size(); ++i) {
    int conf_ch = yolo_v2_handler_->GetAnchorChannel(i, YOLOV2Handler<Dtype>::AnchorChannel::CONF);
    Dtype conf = bgm::Sigmoid<Dtype>(*(cell_data_[conf_ch]));

    std::vector<Dtype> cls_softmax;
    int cls_idx;
    ClsSoftmax(i, &cls_softmax, &cls_idx); 

    conf *= anchor_weight_[i];
    //conf *= cls_softmax[cls_idx];

    if (conf > conf_threshold) {
      cv::Rect_<Dtype> yolo_box = GetYOLOBox(i);
      cv::Rect_<Dtype> raw_box = yolo_v2_handler_->YOLOBoxToRawBox(yolo_box, h, w, i);

      detection->push_back({cls_idx + 1, raw_box, static_cast<float>(conf)});
    }
  }
}

template <typename Dtype>
void YOLOV2SoftmaxDecoder<Dtype>::ClsSoftmax(
    int anchor_idx, std::vector<Dtype>* cls_conf, int* max_idx) const {
  CHECK(cls_conf);
  CHECK(max_idx);

  cls_conf->resize(yolo_v2_handler_->num_class());

  int idx = yolo_v2_handler_->GetAnchorChannel(anchor_idx, YOLOV2Handler<Dtype>::AnchorChannel::CLASS_BEGIN);
  Dtype exp_sum = 0;
  for (int i = 0; i < cls_conf->size(); ++i) {
    (*cls_conf)[i] = std::exp(*(cell_data_[idx + i]));
    exp_sum += (*cls_conf)[i];
  }

  *max_idx = 0;
  Dtype max_conf = (*cls_conf)[0];
  for (int i = 0; i < cls_conf->size(); ++i) {
    if ((*cls_conf)[i] > max_conf) {
      max_conf = (*cls_conf)[i];
      *max_idx = i;
    }

    (*cls_conf)[i] /= exp_sum;
  }
}
} // namespace bgm

#endif // !BGM_YOLO_V2_DECODER_HPP_
