#include "label_remap_layer.hpp"

#include "caffe/layer_factory.hpp"

namespace caffe
{

template <typename Dtype>
void LabelRemapLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[0]->height(), 1);

  InitRemapTable(this->layer_param().label_remap_param());
}

template <typename Dtype>
void LabelRemapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                 const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); i++) {
    int size = bottom[i]->count();
    const Dtype* src_iter = bottom[i]->cpu_data();
    Dtype* dst_iter = top[i]->mutable_cpu_data();
    for (int j = size; j--; )
      *dst_iter++ = MapLabel(*src_iter++);
  }
}

template <typename Dtype>
void LabelRemapLayer<Dtype>::InitRemapTable(
    const LabelRemapParameter& param) {
  CHECK_EQ(param.src_size(), param.dst_size());

  label_remap_table_.clear();

  for (int i = 0; i < param.src_size(); i++) {
    int src = param.src().Get(i);
    int dst = param.dst().Get(i);
    label_remap_table_.insert(std::make_pair(src, dst));
  }
}

#ifdef CPU_ONLY
STUB_GPU(LabelRemapLayer);
#endif

INSTANTIATE_CLASS(LabelRemapLayer);
REGISTER_LAYER_CLASS(LabelRemap);

} // namespace caffe