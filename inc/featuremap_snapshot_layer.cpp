#include "featuremap_snapshot_layer.hpp"

#include "caffe/util/format.hpp"

namespace caffe
{
template <typename Dtype>
void FeaturemapSnapshotLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const FeaturemapSnapshotParameter& param = this->layer_param_.featuremap_snapshot_param();

  if (param.type() == FeaturemapSnapshotParameter::LMDB) {
    db_.reset(caffe::db::GetDB("lmdb"));
    db_->Open(param.out_dst(), caffe::db::NEW);
    txn_.reset(db_->NewTransaction());
    txn_count_ = 0;
  }
  else {
    LOG(FATAL) << "Not implemented yet";
  }
}

template <typename Dtype>
void FeaturemapSnapshotLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BlobProtoVector blob_vec;
  blob_vec.clear_blobs();

  for(int i=0; i<bottom.size(); ++i)
    bottom[i]->ToProto(blob_vec.add_blobs(), false);

  std::string key_str = caffe::format_int(txn_count_, 8);
  std::string data_str;
  blob_vec.SerializeToString(&data_str);
  txn_->Put(key_str, data_str);
  
  if (++txn_count_ % COMMIT_PERIOD == 0) {
    txn_->Commit();
    txn_.reset(db_->NewTransaction());
  }
}

#ifdef CPU_ONLY
STUB_GPU(FeaturemapSnapshotLayer);
#endif

INSTANTIATE_CLASS(FeaturemapSnapshotLayer);
REGISTER_LAYER_CLASS(FeaturemapSnapshot);
} // namespace caffe