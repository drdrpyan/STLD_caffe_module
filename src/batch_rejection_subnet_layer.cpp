#include "batch_rejection_subnet_layer.hpp"

namespace caffe
{

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const BatchRejectionParameter& param = this->layer_param_.batch_rejection_param();
  
  rejection_by_threshold_ = param.rejection_by_threshold();
  if (rejection_by_threshold_)
    threshold_ = param.threshold();

  CHECK(bottom.size() == top.size() || bottom.size() == top.size() + 1);

  std::vector<Blob<Dtype>*> subnet_bottom, subnet_top;
  GetSubnetBottomTop(bottom, top, &subnet_bottom, &subnet_top);

  SubnetLayer<Dtype>::LayerSetUp(subnet_bottom, subnet_top);
}

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK(bottom.size() == top.size() || bottom.size() == top.size() + 1);

  const Blob<Dtype>& rejection_info = *(bottom.back());
  int num_batch = bottom[0]->num();
  for (int i = 1; i < bottom.size() - 1; ++i)
    CHECK_EQ(bottom[i]->num(), num_batch);
  if (rejection_by_threshold_)
    CHECK_EQ(bottom.back()->count(), num_batch);
  else
    CHECK_LE(bottom.back()->count(), num_batch);

  std::vector<Blob<Dtype>*> subnet_bottom, subnet_top;
  GetSubnetBottomTop(bottom, top, &subnet_bottom, &subnet_top);

  //SubnetLayer<Dtype>::Reshape(subnet_bottom, subnet_top);
  const std::vector<Blob<Dtype>*>& net_output = net_->output_blobs();
  for (int i = 0; i < subnet_top.size(); ++i)
    subnet_top[i]->ReshapeLike(*(net_output[i]));

  if (bottom.size() == top.size()) {
    std::vector<int> top_shape(1, num_batch);
    top.back()->Reshape(top_shape);
  }
}

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward(false, bottom, top);
}

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward(true, bottom, top);
}

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::Forward(
    bool copy_gpu,
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<int> rejection_idx;
  GetRejectionIdx(*(bottom.back()), &rejection_idx);

  std::vector<Blob<Dtype>*> subnet_bottom, subnet_top;
  GetSubnetBottomTop(bottom, top, &subnet_bottom, &subnet_top);

  if (rejection_idx.size() == bottom[0]->num())
    ForwardEmpty(copy_gpu, subnet_top);
  else {
    ForwardSubnet(copy_gpu, rejection_idx, subnet_bottom, subnet_top);

    if (bottom.size() == top.size()) {
      if (rejection_by_threshold_)
        ForwardRejectionInfo(rejection_idx, bottom.back());
      else
        top.back()->CopyFrom(*(bottom.back()), false, true);
    }
  }
}

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::ForwardEmpty(
    bool copy_gpu, const std::vector<Blob<Dtype>*>& top) const {
  if (copy_gpu)
    for (int i = 0; i < top.size(); ++i)
#ifdef CPU_ONLY
      caffe_set(top[i]->count(), static_cast<Dtype>(0),
                top[i]->mutable_cpu_data());
#else
      cudaMemset(top[i]->mutable_gpu_data(), 0,
                 sizeof(Dtype)*(top[i]->count()));
#endif
  else
    for (int i = 0; i < top.size(); ++i)
      caffe_set(top[i]->count(), static_cast<Dtype>(0),
                top[i]->mutable_cpu_data());
}

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::ForwardSubnet(
    bool copy_gpu,
    const std::vector<int>& rejection_idx,
    const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  CHECK_GT(rejection_idx.size(), 0);
  CHECK_LT(rejection_idx.size(), bottom[0]->num());
  CHECK_EQ(bottom.size(), top.size());

  const std::vector<Blob<Dtype>*>& subnet_input = net_->input_blobs();
  for (int i = 0; i < bottom.size(); ++i)
    RejectBatch(*(bottom[i]), rejection_idx,
                copy_gpu, subnet_input[i]);

  net_->Forward();

  const std::vector<Blob<Dtype>*>& subnet_output = net_->output_blobs();
  for (int i = 0; i < subnet_output.size(); ++i)
    InsertDummyBatch(*(subnet_output[i]), rejection_idx, 
                      copy_gpu, top[i]);
}

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::ForwardRejectionInfo(
    const std::vector<int>& rejection_idx, Blob<Dtype>* top) const {
  CHECK(top);
  std::vector<int> top_shape(1, rejection_idx.size());
  top->Reshape(top_shape);

  Dtype* top_data = top->mutable_cpu_data();
  for (int i = 0; i < rejection_idx.size(); ++i)
    top_data[i] = rejection_idx[i];
}

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::GetSubnetBottomTop(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    vector<Blob<Dtype>*>* subnet_bottom,
    vector<Blob<Dtype>*>* subnet_top) const {
  CHECK(subnet_bottom);
  CHECK(subnet_top);

  subnet_bottom->assign(bottom.begin(), bottom.end() - 1);
  subnet_top->assign(top.begin(), top.begin() + bottom.size() - 1);
}

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::GetRejectionIdx(
    const Blob<Dtype>& rejection_info, std::vector<int>* rejection_idx) const {
  CHECK(rejection_idx);

  rejection_idx->clear();

  const Dtype* rejection_info_data = rejection_info.cpu_data();

  if (rejection_by_threshold_) {
    for (int i = 0; i < rejection_info.count(); ++i)
      if (rejection_info_data[i] > threshold_)
        rejection_idx->push_back(i);
  }
  else {
    rejection_idx->resize(rejection_info.count());
    for (int i = 0; i < rejection_info.count(); ++i) {
      CHECK_GE(rejection_info_data[i], 0);
      CHECK_LT(rejection_info_data[i], rejection_info.count());
      (*rejection_idx)[i] = rejection_info_data[i];
    }
  }
}

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::RejectBatch(
    const Blob<Dtype>& src, const std::vector<int>& rejection_idx,
    bool copy_gpu, Blob<Dtype>* dst) const {
  CHECK(!rejection_idx.empty());
  CHECK_LT(rejection_idx.size(), src.num());
  CHECK(dst);

  std::vector<int> dst_shape = src.shape();
  dst_shape[0] = src.num() - rejection_idx.size();
  dst->Reshape(dst_shape);

  const Dtype* src_ptr = copy_gpu ? src.gpu_data() : src.cpu_data();
  Dtype* dst_iter = copy_gpu ? dst->mutable_gpu_data() : dst->mutable_cpu_data();
  const int BATCH_ELEMS = src.count(1);

  std::vector<int> sorted_rejection_idx(rejection_idx);
  std::sort(sorted_rejection_idx.begin(), sorted_rejection_idx.end());
  int idx = 0;
  for (int i = 0; i < src.num(); ++i) {
    if (i != sorted_rejection_idx[idx]) {
      caffe_copy<Dtype>(BATCH_ELEMS, src_ptr + src.offset(i),
                        dst_iter);
      dst_iter += BATCH_ELEMS;
    }
    else
      ++idx;
  }
}

template <typename Dtype>
void BatchRejectionSubnetLayer<Dtype>::InsertDummyBatch(
    const Blob<Dtype>& src, const std::vector<int> rejection_idx,
    bool copy_gpu, Blob<Dtype>* dst) {
  CHECK_GT(src.num(), 0);
  CHECK(!rejection_idx.empty());
  CHECK(dst);

  std::vector<int> dst_shape = src.shape();
  dst_shape[0] = src.num() + rejection_idx.size();
  dst->Reshape(dst_shape);

  const Dtype* src_iter = copy_gpu ? src.gpu_data() : src.cpu_data();
  Dtype* dst_ptr = copy_gpu ? dst->mutable_gpu_data() : dst->mutable_cpu_data();
  const int BATCH_ELEMS = src.count(1);

  std::vector<int> sorted_rejection_idx(rejection_idx);
  std::sort(sorted_rejection_idx.begin(), sorted_rejection_idx.end());
  int idx = 0;
  for (int i = 0; i < dst->num(); ++i) {
    if (i != sorted_rejection_idx[idx]) {
      caffe_copy<Dtype>(BATCH_ELEMS, src_iter, dst_ptr + dst->offset(i));
      src_iter += BATCH_ELEMS;
    }
    else {
      caffe_set<Dtype>(BATCH_ELEMS, static_cast<Dtype>(0),
                       dst_ptr + dst->offset(i));
      ++idx;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BatchRejectionSubnetLayer);
#endif

INSTANTIATE_CLASS(BatchRejectionSubnetLayer);
REGISTER_LAYER_CLASS(BatchRejectionSubnet);

}