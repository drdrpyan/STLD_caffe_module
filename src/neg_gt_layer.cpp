#include "neg_gt_layer.hpp"

namespace caffe
{
template <typename Dtype>
void NegGTLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  const NegGTParameter& param = this->layer_param().neg_gt_param();

  batch_size_ = param.batch_size();
  CHECK_GT(batch_size_, 0);

  num_gt_ = param.num_gt();
  CHECK_GT(num_gt_, 0);
}

template <typename Dtype>
void NegGTLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top) {
  std::vector<int> top_shape(3);
  top_shape[0] = batch_size_;
  top_shape[1] = 1;
  top_shape[2] = (bottom.size() > 0) ? bottom[0]->height() : num_gt_;

  top[0]->Reshape(top_shape);

  if (top.size() > 1) {
    top_shape[1] = 4;
    top_shape[2] = (bottom.size() > 1) ? bottom[1]->height() : num_gt_;
    top[1]->Reshape(top_shape);
  }
}

template <typename Dtype>
void NegGTLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  Dtype* label_data = top[0]->mutable_cpu_data();
  //caffe_set<Dtype>(top[0]->count(), LabelParameter::NONE, label_data);
  caffe_set<Dtype>(top[0]->count(), LabelParameter::DUMMY_LABEL, label_data);

  if (top.size() > 1) {
    Dtype* bbox_data = top[1]->mutable_cpu_data();
    caffe_set<Dtype>(top[1]->count(), BBoxParameter::DUMMY_VALUE, bbox_data);
  }
}

template <typename Dtype>
void NegGTLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
  //Dtype* label_data = top[0]->mutable_gpu_data();
  //caffe_set<Dtype>(top[0]->count(), LabelParameter::NONE, label_data);

  //if (top.size() > 1) {
  //  Dtype* bbox_data = top[1]->mutable_gpu_data();
  //  caffe_set<Dtype>(top[1]->count(), BBoxParameter::DUMMY_VALUE, bbox_data);
  //}
}


INSTANTIATE_CLASS(NegGTLayer);
REGISTER_LAYER_CLASS(NegGT);
} // namespace caffe