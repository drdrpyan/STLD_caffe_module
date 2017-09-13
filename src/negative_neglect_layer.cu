#include "negative_neglect_layer.hpp"

namespace caffe
{

//template <typename Dtype>
//void NegativeNeglectLayer<Dtype>::Forward_gpu(
//    const vector<Blob<Dtype>*>& bottom,
//    const vector<Blob<Dtype>*>& top) {
//  const Blob<Dtype>& predicted_bbox = *(bottom[0]);
//  const Blob<Dtype>& label = *(bottom[1]);
//  Blob<Dtype>& out = *(top[0]);
//
//  out.CopyFrom(predicted_bbox);
//
//  const int HW = label.height() * label.width();
//
//  for (int n = 0; n < label.num(); n++) {
//    for (int h = 0; h < label.height(); h++) {
//      for (int w = 0; w < label.width(); w++) {
//        const Dtype* label_iter = label.gpu_data() + label.offset(n, 0, h, w);
//        Dtype* out_iter = out.mutable_gpu_data() + out.offset(n, 0, h, w);
//        for (int c = 0; c < label.channels(); c++) {
//          if (*label_iter == LabelParameter::DUMMY_LABEL ||
//              *label_iter == LabelParameter::NONE) {
//            for (int d = 0; d < bbox_dim_; d++)
//              *(out_iter + d * HW) = BBoxParameter::DUMMY_VALUE;
//          }
//
//          out_iter += bbox_dim_ * HW;
//        }
//      }
//    }
//  }
//  
//}


//INSTANTIATE_LAYER_GPU_FUNCS(NegativeNeglectLayer);

} // namespace caffe