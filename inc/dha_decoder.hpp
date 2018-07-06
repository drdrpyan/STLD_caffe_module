#ifndef BGM_DHA_DECODER_HPP_
#define BGM_DHA_DECODER_HPP_

#include "detection_util.hpp"
#include "dha_handler.hpp"
#include "detection_nms.hpp"

#include "caffe/blob.hpp"

#include <algorithm>

namespace bgm
{

template <typename Dtype>
class DHADecoder
{
 public:
  DHADecoder(DHAHandler<Dtype>* dha_handler,
             DetectionNMS<Dtype>* nms,
             const std::vector<float>& anchor_weight = std::vector<float>(),
             const std::vector<float>& class_weight = std::vector<float>(),
             bool class_label_zero_begin = false);
  virtual void Decode(
      const caffe::Blob<Dtype>& dha_out, 
      Dtype neg_threshold, bool bbox_nms,
      std::vector<std::vector<Detection<Dtype> > >* detection);

 protected:
  void InitCellData(const caffe::Blob<Dtype>& dha_out, int batch_idx);
  virtual bool DecodeCellData(
      int h, int w, Dtype neg_threshold, Detection<Dtype>* detection) const;
  void DecodeAnchorConf(std::vector<Dtype>* pos_conf,
                        Dtype* neg_conf) const;
  void DecodeClassConf(int anchor, std::vector<Dtype>* class_conf) const;
  cv::Rect_<Dtype> DecodeBBox(int anchor, int h, int w) const;

  void NextCellData();

  DetectionNMS<Dtype>& nms();

  //float neg_threshold_;
  bool class_label_zero_begin_;
  std::vector<float> anchor_weight_;
  std::vector<float> class_weight_;

  std::vector<const Dtype*> cell_data_;

  std::unique_ptr<DHAHandler<Dtype> > dha_handler_;
  std::unique_ptr<DetectionNMS<Dtype> > nms_;
};

// template fucntions
template <typename Dtype>
DHADecoder<Dtype>::DHADecoder(DHAHandler<Dtype>* dha_handler,
                              DetectionNMS<Dtype>* nms,
                              const std::vector<float>& anchor_weight,
                              const std::vector<float>& class_weight,
                              bool class_label_zero_begin) 
  : dha_handler_(dha_handler), nms_(nms),
    anchor_weight_(anchor_weight), class_weight_(class_weight),
    class_label_zero_begin_(class_label_zero_begin) {
  if (anchor_weight_.empty())
    anchor_weight_.resize(dha_handler_->anchor().size(), 1.0f);
  if (class_weight_.empty())
    class_weight_.resize(dha_handler_->num_class(), 1.0f);

  CHECK_EQ(anchor_weight_.size(), dha_handler_->anchor().size());
  CHECK_EQ(class_weight_.size(), dha_handler_->num_class());
}

template <typename Dtype>
void DHADecoder<Dtype>::Decode(
    const caffe::Blob<Dtype>& dha_out,
    Dtype neg_threshold, bool bbox_nms,
    std::vector<std::vector<Detection<Dtype> > >* detection) {
  CHECK(detection);
  detection->resize(dha_out.num());

  for (int n = 0; n < dha_out.num(); ++n) {
    InitCellData(dha_out, n);
    std::vector<Detection<Dtype> >& batch_detection = (*detection)[n];
    batch_detection.clear();

    for (int h = 0; h < dha_out.height(); ++h) {
      for (int w = 0; w < dha_out.width(); ++w) {
        Detection<Dtype> cell_detection;
        bool detected = DecodeCellData(h, w, neg_threshold, 
                                       &cell_detection);

        if (detected)
          batch_detection.push_back(cell_detection);

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
void DHADecoder<Dtype>::InitCellData(
    const caffe::Blob<Dtype>& dha_out, int batch_idx) {
  cell_data_.resize(dha_out.channels());
  
  const Dtype* ptr = dha_out.cpu_data() + dha_out.offset(batch_idx);
  int step = dha_out.count(2);
  for (auto iter = cell_data_.begin(); iter != cell_data_.end(); ++iter) {
    *iter = ptr;
    ptr += step;
  }
}

template <typename Dtype>
bool DHADecoder<Dtype>::DecodeCellData(
    int h, int w, Dtype neg_threshold, Detection<Dtype>* detection) const {
  CHECK(detection);

  std::vector<Dtype> pos_conf;
  Dtype neg_conf;
  DecodeAnchorConf(&pos_conf, &neg_conf);

  if (neg_conf > neg_threshold) {
    return false;
  }
  else {
    for (int i = 0; i < pos_conf.size(); ++i)
      pos_conf[i] *= anchor_weight_[i];
    int anchor_idx = std::distance(pos_conf.cbegin(),
                                   std::max_element(pos_conf.cbegin(),
                                                    pos_conf.cend()));

    std::vector<Dtype> class_conf;
    DecodeClassConf(anchor_idx, &class_conf);
    for (int i = 0; i < dha_handler_->num_class(); ++i)
      class_conf[i] *= class_weight_[i];
    int max_class = std::distance(class_conf.cbegin(),
                                  std::max_element(class_conf.cbegin(),
                                                   class_conf.cend()));
    if (!class_label_zero_begin_)
      ++max_class;

    cv::Rect_<Dtype> bbox = DecodeBBox(anchor_idx, h, w);

    detection->conf = pos_conf[anchor_idx];
    detection->label = max_class;
    detection->bbox = bbox;

    return true;
  }
}

template <typename Dtype>
void DHADecoder<Dtype>::DecodeAnchorConf(std::vector<Dtype>* pos_conf,
                                         Dtype* neg_conf) const {
  CHECK(pos_conf);
  CHECK(neg_conf);

  std::vector<Dtype> raw_conf(dha_handler_->anchor().size() + 1);
  for (int i = 0; i < dha_handler_->anchor().size(); ++i) {
    int conf_ch = dha_handler_->GetPosAnchorConfChannel(i);
    raw_conf[i] = *(cell_data_[conf_ch]);
  }
  raw_conf.back() = *(cell_data_[dha_handler_->GetNegAnchorConfChannel()]);

  std::vector<Dtype> softmax;
  bgm::Softmax(raw_conf, &softmax);

  //pos_conf->resize(raw_conf.size() - 1);
  //for (int i = 0; i < pos_conf->size(); ++i)
  //  (*pos_conf)[i] = softmax[i];
  pos_conf->assign(softmax.cbegin(), softmax.cend() - 1);

  *neg_conf = softmax.back();
}

template <typename Dtype>
void DHADecoder<Dtype>::DecodeClassConf(
    int anchor, std::vector<Dtype>* class_conf) const {
  CHECK(class_conf);

  std::vector<Dtype> raw_conf(dha_handler_->num_class());
  for (int i = 0; i < raw_conf.size(); ++i) {
    int conf_ch = dha_handler_->GetClassChannel(anchor, i, true);
    raw_conf[i] = *(cell_data_[conf_ch]);
  }

  bgm::Softmax(raw_conf, class_conf);
}

template <typename Dtype>
cv::Rect_<Dtype> DHADecoder<Dtype>::DecodeBBox(
    int anchor, int h, int w) const {
  cv::Rect_<Dtype> box;
  box.x = *(cell_data_[dha_handler_->GetAnchorChannel(anchor, DHAHandler<Dtype>::AnchorElem::X)]);
  box.y = *(cell_data_[dha_handler_->GetAnchorChannel(anchor, DHAHandler<Dtype>::AnchorElem::Y)]);
  box.width = *(cell_data_[dha_handler_->GetAnchorChannel(anchor, DHAHandler<Dtype>::AnchorElem::W)]);
  box.height = *(cell_data_[dha_handler_->GetAnchorChannel(anchor, DHAHandler<Dtype>::AnchorElem::H)]);

  return dha_handler_->NormalizedBoxToRawBox(box, h, w, anchor);
}

template <typename Dtype>
inline void DHADecoder<Dtype>::NextCellData() {
  for (auto iter = cell_data_.begin(); iter != cell_data_.end(); ++iter)
    ++(*iter);
}

template <typename Dtype>
inline DetectionNMS<Dtype>& DHADecoder<Dtype>::nms() {
  return *nms_;
}
} // namespace bgm

#endif // !BGM_DHA_DECODER_HPP_
