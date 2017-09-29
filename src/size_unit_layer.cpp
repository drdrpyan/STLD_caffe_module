#include "size_unit_layer.hpp"

namespace caffe
{

template <typename Dtype>
void SizeUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  const Dtype* src_iter = bottom[0]->cpu_data();
  Dtype* dst_iter = top[0]->mutable_cpu_data();
  for (int i = bottom[0]->count(); i--; ) {
    // *dst_iter++ = std::ceil((*src_iter++) / UNIT_);
    if (*src_iter != BBoxParameter::DUMMY_VALUE)
      *dst_iter = std::ceil((*src_iter) / UNIT_);
    else
      *dst_iter = BBoxParameter::DUMMY_VALUE;

    ++src_iter;
    ++dst_iter;
  }
}

INSTANTIATE_CLASS(SizeUnitLayer);
REGISTER_LAYER_CLASS(SizeUnit);

} // namespace caffe