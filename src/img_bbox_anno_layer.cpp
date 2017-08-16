#include "img_bbox_anno_layer.hpp"

#include "caffe_extend.pb.h"

#include "caffe/data_transformer.hpp"
#include "caffe/util/benchmark.hpp"

#include <vector>

using caffe_ext::ImgBBoxAnnoDatum;
using caffe::Batch;
using std::vector;

namespace bgm
{

template <typename Dtype>
ImgBBoxAnnoLayer<Dtype>::ImgBBoxAnnoLayer(
    const caffe_ext::ExtendedLayerParameter& param) 
  : caffe::DataLayer<Dtype>(param.layer_param()){

}
template <typename Dtype>
void ImgBBoxAnnoLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();
  ImgBBoxAnnoDatum img_bbox_anno_datum;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }
    img_bbox_anno_datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();
    batch->data_.offset()

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = 
        this->data_transformer_->InferBlobShape(img_bbox_anno_datum.img_datum());
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }
    
    // 여기에 남은 영역 추가할 것
    //batch
    //batch->data_.

    timer.Start();
    trans_time += timer.MicroSeconds();
    Next();
  }

  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void ImgBBoxAnnoLayer<Dtype>::CopyImage(
    int item_id,
    const caffe_ext::ImgBBoxAnnoDatum& datum,
    caffe::Blob<Dtype>* batch_data) const {
  int offset = batch_data->offset(item_id);
  Dtype* top_data = batch_data->mutable_cpu_data();
  //std::copy(datum.img_datum().b, top_data+offset)
}
//INSTANTIATE_CLASS(ImgBBoxAnnoLayer);
//REGISTER_LAYER_CLASS(ImgBBoxAnno);

} // namespace bgm