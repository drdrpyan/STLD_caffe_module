#include "featuremap_data_layer.hpp"

#include "caffe/util/benchmark.hpp"

namespace caffe
{

template <typename Dtype>
void FeaturemapDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();

  BlobProtoVector blob_proto_vec;
  blob_proto_vec.ParseFromString(cursor_->value());

  Blob<Dtype> featuremap_blob;
  featuremap_blob.FromProto(blob_proto_vec.blobs(0));
  std::vector<int> top_shape(featuremap_blob.shape());
  top_shape[0] *= batch_size;
  top[0]->Reshape(top_shape);

  if (this->output_labels_) {
    Blob<Dtype> gt_blob;
    gt_blob.FromProto(blob_proto_vec.blobs(1));
    std::vector<int> top_shape(gt_blob.shape());
    top_shape[0] *= batch_size;
    top[1]->Reshape(top_shape);
  }
}

template <typename Dtype>
void FeaturemapDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  //CPUTimer batch_timer;
  //batch_timer.Start();
  //double read_time = 0;
  //double trans_time = 0;
  //CPUTimer timer;
  //CHECK(batch->data_.count());
  //CHECK(this->transformed_data_.count());
  //const int batch_size = this->layer_param_.data_param().batch_size();

  //BlobProtoVector blob_proto_vec;
  //for (int item_id = 0; item_id < batch_size; ++item_id) {
  //  timer.Start();
  //  while (Skip()) {
  //    Next();
  //  }
  //  blob_proto_vec.ParseFromString(cursor_->value());
  //  read_time += timer.MicroSeconds();

  //  if (item_id == 0) {
  //    // Reshape according to the first datum of each batch
  //    // on single input batches allows for inputs of varying dimension.
  //    // Use data_transformer to infer the expected blob shape from datum.
  //    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  //    this->transformed_data_.Reshape(top_shape);
  //    // Reshape batch according to the batch_size.
  //    top_shape[0] = batch_size;
  //    batch->data_.Reshape(top_shape);
  //  }

  //  // Apply data transformations (mirror, scale, crop...)
  //  timer.Start();
  //  int offset = batch->data_.offset(item_id);
  //  Dtype* top_data = batch->data_.mutable_cpu_data();
  //  this->transformed_data_.set_cpu_data(top_data + offset);
  //  this->data_transformer_->Transform(datum, &(this->transformed_data_));
  //  // Copy label.
  //  if (this->output_labels_) {
  //    Dtype* top_label = batch->label_.mutable_cpu_data();
  //    top_label[item_id] = datum.label();
  //  }
  //  trans_time += timer.MicroSeconds();
  //  Next();
  //}
  //timer.Stop();
  //batch_timer.Stop();
  //DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

} // namespace caffe