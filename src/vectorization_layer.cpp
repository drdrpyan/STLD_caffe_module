#include "vectorization_layer.hpp"

namespace caffe
{

template <typename Dtype>
void VectorizationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); i++) {
    std::vector<int> top_shape;
    ComputeTopShape(*(bottom[i]), &top_shape);
    top[i]->Reshape(top_shape);
  }
}

template <typename Dtype>
void VectorizationLayer<Dtype>::ComputeTopShape(
    const std::vector<int>& bottom_shape,
    std::vector<int>* top_shape) const {
  CHECK(top_shape);

  top_shape->resize(4);
  (*top_shape)[0] = bottom_shape[0] * bottom_shape[2] * bottom_shape[3];
  (*top_shape)[1] = bottom_shape[1];
  (*top_shape)[2] = 1;
  (*top_shape)[4] = 1;
}

template <typename Dtype>
void VectorizationLayer<Dtype>::Vectorize_cpu(
    const Blob<Dtype>& bottom, Blob<Dtype>* top) const {
  CHECK(top);

  Dtype* top_iter = top->mutable_cpu_data();
  for (int bot_n = 0; bot_n < bottom.num(); bot_n++) {
    std::vector<const Dtype*> ch_iters;
    GetDataChIters(bottom, bot_n, &ch_iters);

    for (i = bottom.height() * bottom.width(); i--; )
      for (j = 0; j < bottom.channels(); j++)
        *top_iter++ = *(ch_iters[j]++);
  }
}

template <typename Dtype>
void VectorizationLayer<Dtype>::Devectorize_cpu(
    const Blob<Dtype>& top, Blob <Dtype> *bottom) const {
  CHECK(bottom);
  
  const Dtype* top_iter = top->cpu_data();
  for (int bot_n = 0; bot_n < bottom->num(); bot_n++) {
    std::vector<Dtype*> ch_iters;
    GetDataChIters(bottom, bot_n, &ch_iters);
    for (i = bottom.height() * bottom.width(); i--; )
      for (j = 0; j < bottom.channels(); j++)
        (*ch_iters[j]++) = *top_iter++;
  }
}

template <typename Dtype>
void VectorizationLayer<Dtype>::GetDataChIters(
    const Blob<Dtype>& blob, int n,
    std::vector<const Dtype*>* ch_iters) const {
  CHECK(n >= 0 && n < blob.num());
  CHECK(ch_iters);

  ch_iters->resize(blob.channels());
  (*ch_iters)[0] = blob.cpu_data() + blob.offset(n);
  int hw = blob.height() * blob.width();
  for (int i = 1; i < ch_iters->size(); i++)
    (*ch_iters)[i] = (*ch_iters)[i - 1] + hw;
}

template <typename Dtype>
void VectorizationLayer<Dtype>::GetDiffChIters(
    const Blob<Dtype>& blob, int n,
    std::vector<Dtype*>* ch_iters) const {
    CHECK(n >= 0 && n < blob.num());
  CHECK(ch_iters);
  
  ch_iters->resize(blob.channels());
  (*ch_iters)[0] = blob.mutable_diff_data() + blob.offset(n);
  int hw = blob.height() * blob.width();
  for (int i = 1; i < ch_iters->size(); i++)
    (*ch_iters)[i] = (*ch_iters)[i - 1] + hw;
}
} // namespace caffe