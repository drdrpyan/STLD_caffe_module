//#ifndef TLR_BGM_YOLO_V2_SUBWIN_RESULT_LAYER_HPP_
//#define TLR_BGM_YOLO_V2_SUBWIN_RESULT_LAYER_HPP_
//
//#include "yolo_v2_result_layer.hpp"
//
//namespace caffe
//{
//
//template <typename Dtype>
//class YOLOV2SubwinResultLayer : public YOLOV2ResultLayer<Dtype>
//{
// public:
//  explicit YOLOV2SubwinResultLayer(const LayerParameter& param);
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//                          const vector<Blob<Dtype>*>& top) override;
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//                       const vector<Blob<Dtype>*>& top) override;
//  virtual const char* type() const override;
//};
//
//// inline functions
//template <typename Dtype>
//const char* YOLOV2SubwinResultLayer<Dtype>::type() const {
//  return "YOLOV2SubwinResult";
//}
//
//} // namespace caffe
//
//#endif // !TLR_BGM_YOLO_V2_SUBWIN_RESULT_LAYER_HPP_