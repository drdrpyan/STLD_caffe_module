#include "negative_neglect_layer.hpp"

#include "caffe/proto/caffe.pb.h"

namespace caffe
{

template <typename Dtype>
void NegativeNeglectLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& predicted_bbox = *(bottom[0]);
  const Blob<Dtype>& label = *(bottom[1]);

  CHECK_EQ(label.num(), predicted_bbox.num());
  CHECK_EQ(label.height(), predicted_bbox.height());
  CHECK_EQ(label.width(), predicted_bbox.width());
  //CHECK_GT(label.channels(), 0); // ?
  CHECK_EQ(label.channels(), 1);
  //CHECK_LE(label.channels(), predicted_bbox.channels()); // ?
  //CHECK_EQ(predicted_bbox.channels() % label.channels(), 0); // ?

  //bbox_dim_ = predicted_bbox.channels() / label.channels();  // ?
}

template <typename Dtype>
void NegativeNeglectLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& predicted_bbox = *(bottom[0]);
  const Blob<Dtype>& label = *(bottom[1]);
  Blob<Dtype>& out = *(top[0]);

  out.CopyFrom(predicted_bbox);

  const int HW = label.height() * label.width();

  for (int n = 0; n < label.num(); n++) {
    for (int h = 0; h < label.height(); h++) {
      for (int w = 0; w < label.width(); w++) {
        const Dtype* label_iter = label.cpu_data() + label.offset(n, 0, h, w);
        Dtype* out_iter = out.mutable_cpu_data() + out.offset(n, 0, h, w);
        if (*label_iter == LabelParameter::DUMMY_LABEL ||
            *label_iter == LabelParameter::NONE) {
          for(int c = 0; c<predicted_bbox.channels(); c++)
            *(out_iter + c*HW) = BBoxParameter::DUMMY_VALUE;
        }
        //for (int c = 0; c < label.channels(); c++) {
        //  if (*label_iter == LabelParameter::DUMMY_LABEL ||
        //      *label_iter == LabelParameter::NONE) {
        //    for (int d = 0; d < bbox_dim_; d++)
        //      *(out_iter + d * HW) = BBoxParameter::DUMMY_VALUE;
        //  }

        //  out_iter += bbox_dim_ * HW;
        //}
      }
    }
  }
  
}

#ifdef CPU_ONLY
STUB_GPU(NegativeNeglectLayer);
#endif

INSTANTIATE_CLASS(NegativeNeglectLayer);
REGISTER_LAYER_CLASS(NegativeNeglect);

} // namespace caffe