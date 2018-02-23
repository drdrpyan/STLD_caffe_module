//#include "featuremap_prefetching_data_layer.hpp"
//
//#include "caffe/util/benchmark.hpp"
//
//#include "boost/thread.hpp"
//
//namespace caffe
//{
//template <typename Dtype>
//FeaturemapPrefetchingDataLayer<Dtype>::FeaturemapDataLayer(
//    const LayerParameter& param) 
//  : Layer<Dtype>(param), 
//    prefetch_(param.data_param().prefetch()),
//    prefetch_free_(), prefetch_full_(), prefetch_current_(),
//    offset_() {
//  db_.reset(db::GetDB(param.data_param().backend()));
//  db_->Open(param.data_param().source(), db::READ);
//  cursor_.reset(db_->NewCursor());
//}
//
//template <typename Dtype>
//void FeaturemapPrefetchingDataLayer<Dtype>::LayerSetUp(
//    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//  // init queues 
//  for (int i = 0; i < prefetch_.size(); ++i) {
//    prefetch_[i].reset(new BlobVec(top.size()));
//    prefetch_free_.push(prefetch_[i].get());
//    for (int j = 0; j < prefetch_[i]->size(); ++j) {
//      (*prefetch_[i])[j].mutable_cpu_data();
//#ifndef CPU_ONLY
//      if (Caffe::mode() == Caffe::GPU)
//        (*prefetch_[i])[j].mutable_gpu_data();
//#endif // !CPU_ONLY
//    }
//  }
//
//  // reshape top
//  BlobProtoVector blob_proto_vec;
//  blob_proto_vec.ParseFromString(cursor_->value());
//  CHECK_EQ(blob_proto_vec.blobs_size(), top.size());
//  for (int i = 0; i < blob_proto_vec.blobs_size(); ++i) {
//    Blob<Dtype> temp;
//    temp.FromProto(blob_proto_vec.blobs(i));
//    top[i]->ReshapeLike(temp);
//  }
//
//  StartInternalThread();
//}
//
//template <typename Dtype>
//void FeaturemapPrefetchingDataLayer<Dtype>::InternalThreadEntry() {
//#ifndef CPU_ONLY
//  cudaStream_t stream;
//  if (Caffe::mode() == Caffe::GPU) {
//    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//  }
//#endif
//
//  try {
//    while (!must_stop()) {
//      BlobVec* blob_vec = prefetch_free_.pop();
//      LoadBlobVec(blob_vec);
//#ifndef CPU_ONLY
//      if (Caffe::mode() == Caffe::GPU) {
//        for (int i = 0; i < blob_vec->size(); ++i)
//          (*blob_vec)[i].data().get()->async_gpu_push(stream);
//
//        CUDA_CHECK(cudaStreamSynchronize(stream));
//      }
//#endif
//      prefetch_full_.push(blob_vec);
//    }
//  } catch (boost::thread_interrupted&) {
//    // Interrupted exception is expected on shutdown
//  }
//#ifndef CPU_ONLY
//  if (Caffe::mode() == Caffe::GPU) {
//    CUDA_CHECK(cudaStreamDestroy(stream));
//  }
//#endif
//}
//
//template <typename Dtype>
//void FeaturemapPrefetchingDataLayer<Dtype>::Forward_cpu(
//    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//  if (prefetch_current_) {
//    prefetch_free_.push(prefetch_current_);
//  }
//  prefetch_current_ = prefetch_full_.pop("Waiting for data");
//  CHECK_EQ(prefetch_current_->size(), top.size());
//
//  for (int i = 0; i < top.size(); ++i)
//    top[i]->CopyFrom((*prefetch_current_)[i]);
//}
//
//template <typename Dtype>
//void FeaturemapPrefetchingDataLayer<Dtype>::LoadBlobVec(BlobVec* blob_vec) {
//  CHECK(blob_vec);
//
//  while (Skip()) {
//    Next();
//  }
//  BlobProtoVector blob_proto_vec;
//  blob_proto_vec.ParseFromString(cursor_->value());
//  CHECK_EQ(blob_proto_vec.blobs_size(), blob_vec->size());
//  for (int i = 0; i < blob_proto_vec.blobs_size(); ++i)
//    (*blob_vec)[i].FromProto(blob_proto_vec.blobs(i));
//}
//
//template <typename Dtype>
//bool FeaturemapPrefetchingDataLayer<Dtype>::Skip() {
//  int size = Caffe::solver_count();
//  int rank = Caffe::solver_rank();
//  bool keep = (offset_ % size) == rank ||
//              // In test mode, only rank 0 runs, so avoid skipping
//              this->layer_param_.phase() == TEST;
//  return !keep;
//}
//
//template<typename Dtype>
//void FeaturemapPrefetchingDataLayer<Dtype>::Next() {
//  cursor_->Next();
//  if (!cursor_->valid()) {
//    LOG_IF(INFO, Caffe::root_solver())
//        << "Restarting data prefetching from start.";
//    cursor_->SeekToFirst();
//  }
//  offset_++;
//}
////template <typename Dtype>
////void FeaturemapDataLayer<Dtype>::DataLayerSetUp(
////    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
////  const int batch_size = this->layer_param_.data_param().batch_size();
////
////  BlobProtoVector blob_proto_vec;
////  blob_proto_vec.ParseFromString(cursor_->value());
////
////  Blob<Dtype> featuremap_blob;
////  featuremap_blob.FromProto(blob_proto_vec.blobs(0));
////  std::vector<int> top_shape(featuremap_blob.shape());
////  top_shape[0] *= batch_size;
////  top[0]->Reshape(top_shape);
////
////  if (this->output_labels_) {
////    Blob<Dtype> gt_blob;
////    gt_blob.FromProto(blob_proto_vec.blobs(1));
////    std::vector<int> top_shape(gt_blob.shape());
////    top_shape[0] *= batch_size;
////    top[1]->Reshape(top_shape);
////  }
////}
//
////template <typename Dtype>
////void FeaturemapDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
//  //CPUTimer batch_timer;
//  //batch_timer.Start();
//  //double read_time = 0;
//  //double trans_time = 0;
//  //CPUTimer timer;
//  //CHECK(batch->data_.count());
//  //CHECK(this->transformed_data_.count());
//  //const int batch_size = this->layer_param_.data_param().batch_size();
//
//  //BlobProtoVector blob_proto_vec;
//  //for (int item_id = 0; item_id < batch_size; ++item_id) {
//  //  timer.Start();
//  //  while (Skip()) {
//  //    Next();
//  //  }
//  //  blob_proto_vec.ParseFromString(cursor_->value());
//  //  read_time += timer.MicroSeconds();
//
//  //  if (item_id == 0) {
//  //    // Reshape according to the first datum of each batch
//  //    // on single input batches allows for inputs of varying dimension.
//  //    // Use data_transformer to infer the expected blob shape from datum.
//  //    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
//  //    this->transformed_data_.Reshape(top_shape);
//  //    // Reshape batch according to the batch_size.
//  //    top_shape[0] = batch_size;
//  //    batch->data_.Reshape(top_shape);
//  //  }
//
//  //  // Apply data transformations (mirror, scale, crop...)
//  //  timer.Start();
//  //  int offset = batch->data_.offset(item_id);
//  //  Dtype* top_data = batch->data_.mutable_cpu_data();
//  //  this->transformed_data_.set_cpu_data(top_data + offset);
//  //  this->data_transformer_->Transform(datum, &(this->transformed_data_));
//  //  // Copy label.
//  //  if (this->output_labels_) {
//  //    Dtype* top_label = batch->label_.mutable_cpu_data();
//  //    top_label[item_id] = datum.label();
//  //  }
//  //  trans_time += timer.MicroSeconds();
//  //  Next();
//  //}
//  //timer.Stop();
//  //batch_timer.Stop();
//  //DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
//  //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
//  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
////}
//
////#ifdef CPU_ONLY
////STUB_GPU(FeaturemapPrefetchingDataLayer);
////#endif
////
////INSTANTIATE_CLASS(FeaturemapPrefetchingDataLayer);
////REGISTER_LAYER_CLASS(FeaturemapPrefetchingData);
//
//} // namespace caffe