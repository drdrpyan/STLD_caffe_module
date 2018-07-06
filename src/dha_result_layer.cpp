#include "dha_result_layer.hpp"

#include "dha_subwin_decoder.hpp"

namespace caffe
{

template <typename Dtype>
void DHAResultLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const DHAResultParameter& dha_param = layer_param_.dha_result_param();
  const SubwinOffsetParameter& offset_param = layer_param_.subwin_offset_param();

  num_detection_ = dha_param.num_detection();

  do_nms_ = dha_param.do_nms();

  neg_threshold_ = dha_param.neg_threshold();
  CHECK_GE(neg_threshold_, 0);

  if (layer_param_.has_subwin_offset_param()) {
    global_detection_ = offset_param.global_detection();
    InitDecoder(dha_param, offset_param);
  }
  else {
    LOG(FATAL) << "not implemented yet";
    //global_detection_ = false;
    //InitDecoder(dha_param);
  }

  detection_encoder_.reset(new bgm::DetectionEncoder<Dtype>);
}

template <typename Dtype>
void DHAResultLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<int> top_shape(3);
  top_shape[0] = global_detection_ ? 1 : bottom[0]->num();
  top_shape[1] = 1;
  top_shape[2] = num_detection_;
  top[0]->Reshape(top_shape); // label
  top[2]->Reshape(top_shape); // conf

  top_shape[1] = 4;
  top[1]->Reshape(top_shape); // bbox
}

template <typename Dtype>
void DHAResultLayer<Dtype>::InitDecoder(
    const DHAResultParameter& dha_result_param,
    const SubwinOffsetParameter& subwin_offset_param) {
  cv::Size cell_size(dha_result_param.cell_size().width(),
                     dha_result_param.cell_size().height());
  CHECK_GT(cell_size.width, 0);
  CHECK_GT(cell_size.height, 0);

  int num_class = dha_result_param.num_class();
  CHECK_GT(num_class, 0);

  std::vector<cv::Rect_<Dtype> > anchor(dha_result_param.anchor_size());
  for (int i = 0; i < dha_result_param.anchor_size(); ++i) {
    const caffe::Rect2f& a = dha_result_param.anchor(i);
    anchor[i] = cv::Rect_<Dtype>(a.top_left().x(), a.top_left().y(),
                                 a.size().width(), a.size().height());
  }

  std::vector<float> anchor_weight(dha_result_param.anchor_weight_size());
  if (anchor_weight.size() > 0) {
    for (int i = 0; i < dha_result_param.anchor_weight_size(); ++i)
      anchor_weight[i] = dha_result_param.anchor_weight(i);
    CHECK_EQ(anchor_weight.size(), anchor.size());
  }

  std::vector<float> class_weight(dha_result_param.class_weight_size());
  if (class_weight.size() > 0) {
    for (int i = 0; i < dha_result_param.class_weight_size(); ++i)
      class_weight[i] = dha_result_param.class_weight(i);
    CHECK_EQ(class_weight.size(), num_class);
  }

  float nms_overlap_threshold = dha_result_param.nms_overlap_threshold();
  CHECK_GE(nms_overlap_threshold, 0);

  neg_threshold_ = dha_result_param.neg_threshold();

  bgm::DHAHandler<Dtype>* dha_handler = 
      new bgm::DHAHandler<Dtype>(cell_size, num_class, anchor);
  bgm::DetectionNMS<Dtype>* detection_nms = 
      new bgm::ConfMaxVOCNMS<Dtype>(nms_overlap_threshold);

  std::vector<cv::Point> subwin_offset(subwin_offset_param.win_offset_size());
  for (int i = 0; i < subwin_offset.size(); ++i) {
    subwin_offset[i].x = subwin_offset_param.win_offset(i).x();
    subwin_offset[i].y = subwin_offset_param.win_offset(i).y();
  }
  
  bgm::DHADecoder<Dtype>* decoder = 
      new bgm::DHASubwinDecoder<Dtype>(dha_handler, detection_nms,
                                       anchor_weight, class_weight,
                                       false, 
                                       subwin_offset, 
                                       subwin_offset_param.global_detection());
  dha_decoder_.reset(decoder);
}

#ifdef CPU_ONLY
STUB_GPU(DHAResultLayer);
#endif

INSTANTIATE_CLASS(DHAResultLayer);
REGISTER_LAYER_CLASS(DHAResult);
} // namespace caffe