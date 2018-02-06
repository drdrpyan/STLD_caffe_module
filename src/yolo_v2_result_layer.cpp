#include "yolo_v2_result_layer.hpp"

#include "yolo_v2_subwin_decoder.hpp"

namespace caffe
{
template <typename Dtype>
void YOLOV2ResultLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const YOLOV2ResultParameter& yolo2_param = layer_param_.yolo_v2_result_param();
  const SubwinOffsetParameter& offset_param = layer_param_.subwin_offset_param();
  //cell_size_.width = param.cell_size().width();
  //cell_size_.height = param.cell_size().height();
  //CHECK_GT(cell_size_.width, 0);
  //CHECK_GT(cell_size_.height, 0);

  //num_class_ = param.num_class();
  //CHECK_GT(num_class_, 0);

  //anchor_.resize(param.anchor_size());
  //for (int i = 0; i < param.anchor_size(); ++i) {
  //  const caffe::Rect2f& a = param.anchor(i);
  //  anchor_[i] = cv::Rect_<Dtype>(a.top_left().x(), a.top_left().y(),
  //                                a.size().width(), a.size().height());
  //}

  num_detection_ = yolo2_param.num_detection();

  do_nms_ = yolo2_param.do_nms();

  conf_threshold_ = yolo2_param.conf_threshold();
  CHECK_GE(conf_threshold_, 0);

  if (layer_param_.has_subwin_offset_param()) {
    global_detection_ = offset_param.global_detection();
    InitDecoder(yolo2_param, offset_param);
  }
  else {
    global_detection_ = false;
    InitDecoder(yolo2_param);
  }

  detection_encoder_.reset(new bgm::DetectionEncoder<Dtype>);
}

template <typename Dtype>
void YOLOV2ResultLayer<Dtype>::Reshape(
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
void YOLOV2ResultLayer<Dtype>::InitDecoder(const YOLOV2ResultParameter& param) {
  cv::Size cell_size(param.cell_size().width(), param.cell_size().height());
  CHECK_GT(cell_size.width, 0);
  CHECK_GT(cell_size.height, 0);

  int num_class = param.num_class();
  CHECK_GT(num_class, 0);

  std::vector<cv::Rect_<Dtype> > anchor(param.anchor_size());
  for (int i = 0; i < param.anchor_size(); ++i) {
    const caffe::Rect2f& a = param.anchor(i);
    anchor[i] = cv::Rect_<Dtype>(a.top_left().x(), a.top_left().y(),
                                 a.size().width(), a.size().height());
  }

  float nms_overlap_threshold = param.nms_overlap_threshold();
  CHECK_GE(nms_overlap_threshold, 0);

  bgm::YOLOV2Handler<Dtype>* yolo_v2_handler = new bgm::YOLOV2Handler<Dtype>(cell_size, num_class, anchor);
  //bgm::DetectionNMS<Dtype>* detection_nms = new bgm::VOCNMS<Dtype>(nms_overlap_threshold);
  //bgm::DetectionNMS<Dtype>* detection_nms = new bgm::ConfMaxVOCNMS<Dtype>(nms_overlap_threshold);
  bgm::DetectionNMS<Dtype>* detection_nms = new bgm::MeanSizeNMS<Dtype>(nms_overlap_threshold);
  
  yolo_v2_decoder_.reset(new bgm::YOLOV2Decoder<Dtype>(yolo_v2_handler, detection_nms));
}

template <typename Dtype>
void YOLOV2ResultLayer<Dtype>::InitDecoder(
    const YOLOV2ResultParameter& yolo_v2_result_param,
    const SubwinOffsetParameter& subwin_offset_param) {
  cv::Size cell_size(yolo_v2_result_param.cell_size().width(),
                     yolo_v2_result_param.cell_size().height());
  CHECK_GT(cell_size.width, 0);
  CHECK_GT(cell_size.height, 0);

  int num_class = yolo_v2_result_param.num_class();
  CHECK_GT(num_class, 0);

  std::vector<cv::Rect_<Dtype> > anchor(yolo_v2_result_param.anchor_size());
  for (int i = 0; i < yolo_v2_result_param.anchor_size(); ++i) {
    const caffe::Rect2f& a = yolo_v2_result_param.anchor(i);
    anchor[i] = cv::Rect_<Dtype>(a.top_left().x(), a.top_left().y(),
                                 a.size().width(), a.size().height());
  }

  float nms_overlap_threshold = yolo_v2_result_param.nms_overlap_threshold();
  CHECK_GE(nms_overlap_threshold, 0);

  bgm::YOLOV2Handler<Dtype>* yolo_v2_handler = new bgm::YOLOV2Handler<Dtype>(cell_size, num_class, anchor);
  bgm::DetectionNMS<Dtype>* detection_nms = new bgm::ConfMaxVOCNMS<Dtype>(nms_overlap_threshold);

  std::vector<cv::Point> subwin_offset(subwin_offset_param.win_offset_size());
  for (int i = 0; i < subwin_offset.size(); ++i) {
    subwin_offset[i].x = subwin_offset_param.win_offset(i).x();
    subwin_offset[i].y = subwin_offset_param.win_offset(i).y();
  }
  
  bgm::YOLOV2SubwinDecoder<Dtype>* decoder = 
      new bgm::YOLOV2SubwinDecoder<Dtype>(yolo_v2_handler, detection_nms,
                                          subwin_offset,
                                          subwin_offset_param.global_detection());
  yolo_v2_decoder_.reset(decoder);
}

#ifdef CPU_ONLY
STUB_GPU(YOLOV2ResultLayer);
#endif

INSTANTIATE_CLASS(YOLOV2ResultLayer);
REGISTER_LAYER_CLASS(YOLOV2Result);
}