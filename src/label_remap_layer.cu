#include "label_remap_layer.hpp"

namespace caffe
{

//template <typename Dtype>
//void LabelRemapLayer<Dtype>::Forward_gpu(
//    const vector<Blob<Dtype>*>& bottom,
//    const vector<Blob<Dtype>*>& top) {
//  for (int i = 0; i < bottom.size(); i++) {
//    int size = bottom[i]->count();
//    const Dtype* src_iter = bottom[i]->gpu_data();
//    Dtype* dst_iter = top[i]->mutable_gpu_data();
//    for (int j = size; j--; )
//      *dst_iter++ = MapLabel(*src_iter++);
//  }
//}
//
//INSTANTIATE_LAYER_GPU_FUNCS(LabelRemapLayer);

}