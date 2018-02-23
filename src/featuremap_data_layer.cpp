#include "featuremap_data_layer.hpp"

#include "caffe/util/benchmark.hpp"

#include "boost/thread.hpp"

namespace caffe
{
template <typename Dtype>
FeaturemapDataLayer<Dtype>::FeaturemapDataLayer(
    const LayerParameter& param) 
  : Layer<Dtype>(param) {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
void FeaturemapDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // reshape top
  BlobProtoVector blob_proto_vec;
  blob_proto_vec.ParseFromString(cursor_->value());
  CHECK_EQ(blob_proto_vec.blobs_size(), top.size());
  for (int i = 0; i < blob_proto_vec.blobs_size(); ++i) {
    top[i]->FromProto(blob_proto_vec.blobs(i));
  }
}

template <typename Dtype>
void FeaturemapDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  while (Skip())
    Next();

  BlobProtoVector blob_proto_vec;
  blob_proto_vec.ParseFromString(cursor_->value());
  CHECK_EQ(blob_proto_vec.blobs_size(), top.size());
  for (int i = 0; i < blob_proto_vec.blobs_size(); ++i)
    top[i]->FromProto(blob_proto_vec.blobs(i));

  Next();
}

template <typename Dtype>
bool FeaturemapDataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void FeaturemapDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

#ifdef CPU_ONLY
STUB_GPU(FeaturemapDataLayer);
#endif

INSTANTIATE_CLASS(FeaturemapDataLayer);
REGISTER_LAYER_CLASS(FeaturemapData);

} // namespace caffe