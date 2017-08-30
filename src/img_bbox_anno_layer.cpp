#include "img_bbox_anno_layer.hpp"

//#include "caffe_extend.pb.h"

#include <opencv2/core.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

#include <vector>

using caffe::CPUTimer;
using caffe::Batch;
using caffe::ImgBBoxAnnoDatum;
using std::vector;

namespace caffe
{

//template <typename Dtype>
//ImgBBoxAnnoLayer<Dtype>::ImgBBoxAnnoLayer(
//    const caffe_ext::ExtendedLayerParameter& param) 
//  : caffe::DataLayer<Dtype>(param.layer_param()){
//
//}
template <typename Dtype>
ImgBBoxAnnoLayer<Dtype>::ImgBBoxAnnoLayer(
    const caffe::LayerParameter& param) 
  : caffe::DataLayer<Dtype>(param),
    BATCH_SIZE_(param.data_param().batch_size()),
    IMG_CHANNEL_(
      param.img_bbox_anno_param().colored() ? 3 : 1),
    IMG_HEIGHT_(param.img_bbox_anno_param().img_height()),
    IMG_WIDTH_(param.img_bbox_anno_param().img_width()),
    MAX_NUM_BBOX_(
      param.img_bbox_anno_param().max_bbox_per_img()) {
  CHECK(BATCH_SIZE_ > 0);
  CHECK(IMG_HEIGHT_ > 0);
  CHECK(IMG_WIDTH_ > 0);
  CHECK(MAX_NUM_BBOX_ > 0);
}

template <typename Dtype>
void ImgBBoxAnnoLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //CHECK_EQ(top.size(), 2);

  // Read a data point, and use it to initialize the top blob.
  ImgBBoxAnnoDatum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum.img_datum());
  this->transformed_data_.Reshape(top_shape);

  top_shape[0] = BATCH_SIZE_;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  
  //// reshape top[0] (for images), prefetch_data
  //vector<int> data_shape(4);
  //ComputeDataShape(&data_shape);
  //top[0]->Reshape(data_shape);
  //for (int i = 0; i < this->prefetch_.size(); ++i) {
  //  this->prefetch_[i]->data_.Reshape(data_shape);
  //}

  LOG_IF(INFO, Caffe::root_solver())
    << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();

  // reshape top[1] (for labels)
  if (this->output_labels_) {
    vector<int> label_shape(4);
    ComputeLabelShape(&label_shape);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
}
//template <typename Dtype>
//void ImgBBoxAnnoLayer<Dtype>::Forward_cpu(
//    const vector<Blob<Dtype>*>& bottom,
//    const vector<Blob<Dtype>*>& top) {
//  vector<Blob<Dtype>*> temp_top(2);
//  Blob<Dtype> temp_label;
//  temp_top[0] = top[0];
//  temp_top[1] = &temp_label;
//
//  // call DataLayer::Forward_cpu()
//  caffe::DataLayer<Dtype>::Forward_cpu(bottom, top);
//
//  // split temp_label into label and bbox
//  if (this->output_labels_)
//    ForwardLabelBBox_cpu(temp_label, &top[1], &top[2]);
//}

template <typename Dtype>
void ImgBBoxAnnoLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
  }
}

template <typename Dtype>
void ImgBBoxAnnoLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  //assert(!(this->prefetch_current_));

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  //CHECK(this->transformed_data_.count());
  //const int batch_size = this->layer_param_.data_param().batch_size();

  //ReshpaeBatch(batch);

  ImgBBoxAnnoDatum img_bbox_anno_datum;
  for (int item_id = 0; item_id < BATCH_SIZE_; ++item_id) {
    timer.Start();

    // read datum
    while (Skip())
      Next();
    img_bbox_anno_datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    if (item_id == 0)
      PrepareCopy(img_bbox_anno_datum, batch);
    
    // copy datum
    timer.Start();
    CopyImage(item_id, img_bbox_anno_datum, &(batch->data_));
    trans_time += timer.MicroSeconds();

    if(this->output_labels_)
      CopyLabel(item_id, img_bbox_anno_datum, &(batch->label_));

    Next();
  }

  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void ImgBBoxAnnoLayer<Dtype>::ReshpaeBatch(
    caffe::Batch<Dtype>* batch) const {
  vector<int> batch_data_shape(4);
  //batch_data_shape[0] = BATCH_SIZE_;
  //batch_data_shape[1] = IMG_CHANNEL_;
  //batch_data_shape[2] = IMG_HEIGHT_;
  //batch_data_shape[3] = IMG_WIDTH_;
  //batch->data_.Reshape(batch_data_shape);
  ComputeDataShape(&batch_data_shape);
  batch->data_.Reshape(batch_data_shape);

  if (this->output_labels_) {
    vector<int> batch_label_shape(4);
    //batch_label_shape[0] = BATCH_SIZE_;
    //batch_label_shape[1] = 1;
    //batch_label_shape[0] = MAX_NUM_BBOX_;
    //batch_label_shape[0] = 5; // label, min_x, min_y, max_x, max_y
    //batch->label_.Reshape(batch_label_shape);
    ComputeLabelShape(&batch_label_shape);
    batch->label_.Reshape(batch_label_shape);
  }
}

template <typename Dtype>
void ImgBBoxAnnoLayer<Dtype>::PrepareCopy(
    const caffe::ImgBBoxAnnoDatum& datum,
    caffe::Batch<Dtype>* batch) {
  CHECK(batch);

  std::vector<int> top_shape = 
      this->data_transformer_->InferBlobShape(datum.img_datum());
  this->transformed_data_.Reshape(top_shape);

  top_shape[0] = BATCH_SIZE_;
  batch->data_.Reshape(top_shape);

  if (this->output_labels_) {
    std::vector<int> batch_label_shape;
    ComputeLabelShape(&batch_label_shape);
    batch->label_.Reshape(batch_label_shape);
  }
}


template <typename Dtype>
void ImgBBoxAnnoLayer<Dtype>::CopyImage(
    int item_id,
    const ImgBBoxAnnoDatum& datum,
    caffe::Blob<Dtype>* batch_data) {
  CHECK(batch_data);
  
  Dtype* top_data = batch_data->mutable_cpu_data();
  const int OFFSET= batch_data->offset(item_id);

  //cv::Mat decoded_mat = DecodeDatumToCVMatNative(datum.img_datum());
  Datum decoded_datum;
  decoded_datum.CopyFrom(datum.img_datum());
  DecodeDatumNative(&decoded_datum);

  //std::copy(decoded_datum.data().begin(), 
  //          decoded_datum.data().end(), 
  //          top_data + OFFSET);

  this->transformed_data_.set_cpu_data(top_data + OFFSET);
  this->data_transformer_->Transform(
      decoded_datum, &(this->transformed_data_));
}

template <typename Dtype>
void ImgBBoxAnnoLayer<Dtype>::CopyLabel(
    int item_id,
    const caffe::ImgBBoxAnnoDatum& datum,
    caffe::Blob<Dtype>* batch_label) const {
  CHECK(item_id >= 0 && batch_label);

  const int NUM_BBOX = datum.x_min().size();
  CHECK_LE(NUM_BBOX, MAX_NUM_BBOX_);
  CHECK_EQ(NUM_BBOX, datum.x_max().size());
  CHECK_EQ(NUM_BBOX, datum.y_min().size());
  CHECK_EQ(NUM_BBOX, datum.y_max().size());

  const int OFFSET = batch_label->offset(item_id);
  Dtype *label_itr = batch_label->mutable_cpu_data() + OFFSET;
  Dtype *label_itr2 = label_itr;
  //for (int i = 0; i < MAX_NUM_BBOX_; i++) {
  //  if (i < NUM_BBOX) {
  //    *label_itr++ = static_cast<Dtype>(datum.label(i));
  //    *label_itr++ = static_cast<Dtype>(datum.x_min(i));
  //    *label_itr++ = static_cast<Dtype>(datum.y_min(i));
  //    *label_itr++ = static_cast<Dtype>(datum.x_max(i));
  //    *label_itr++ = static_cast<Dtype>(datum.y_max(i));
  //  }
  //  else {
  //    *label_itr = -1;
  //    *label_itr += 5;
  //  }
  //}
  //for (int i = NUM_BBOX; i--; ) {
  for(int i=0; i<NUM_BBOX; i++) {
    *label_itr++ = static_cast<Dtype>(datum.label(i));
    *label_itr++ = static_cast<Dtype>(datum.x_min(i));
    *label_itr++ = static_cast<Dtype>(datum.y_min(i));
    *label_itr++ = static_cast<Dtype>(datum.x_max(i));
    *label_itr++ = static_cast<Dtype>(datum.y_max(i));
  }
  for (int i = MAX_NUM_BBOX_ - NUM_BBOX; i--; ) {
    *label_itr = caffe::LabelParameter::DUMMY_LABEL;
    label_itr += 5;
  }
}

//template <typename Dtype>
//void ImgBBoxAnnoLayer<Dtype>::FowardLabelBBox_cpu(
//    const Blob<Dtype>& batch_label,
//    Blob<Dtype>* label,
//    Blob<Dtype>* bbox) const {
//  vector<int> label_shape(4);
//  label_shape[0] = BATCH_SIZE_;
//  label_shape[1] = 1;
//  label_shape[2] = 1;
//  label_shape[3] = 1;
//  top[1]->Reshape(label_shape);
//
//  vector<int> bbox_shape(4);
//  bbox_shape[0] = BATCH_SIZE_;
//  bbox_shape[1] = 1;
//  bbox_shape[2] = MAX_NUM_BBOX_;
//  bbox_shape[3] = 4;
//  top[2]->Reshape(bbox_shape);
//
//  Dtype const * batch_label_itr = batch_label.cpu_data();
//  Dtype* label_itr = label->mutable_cpu_data();
//  Dtype* bbox_itr = bbox->mutable_cpu_data();
//
//  for (int i = MAX_NUM_BBOX_; i--; ) {
//    if(batch_label_itr != -1)
//  }
//
//}

template <typename Dtype>
void ImgBBoxAnnoLayer<Dtype>::ComputeDataShape(
    vector<int>* data_shape) const {
  CHECK(data_shape);

  data_shape->resize(4);
  (*data_shape)[0] = BATCH_SIZE_;
  (*data_shape)[1] = IMG_CHANNEL_;
  (*data_shape)[2] = IMG_HEIGHT_;
  (*data_shape)[3] = IMG_WIDTH_;
}

template <typename Dtype>
void ImgBBoxAnnoLayer<Dtype>::ComputeLabelShape(
    vector<int>* label_shape) const {
  CHECK(label_shape);

  label_shape->resize(4);
  (*label_shape)[0] = BATCH_SIZE_;
  (*label_shape)[1] = 1;
  (*label_shape)[2] = MAX_NUM_BBOX_;
  (*label_shape)[3] = 5;
}


INSTANTIATE_CLASS(ImgBBoxAnnoLayer);
REGISTER_LAYER_CLASS(ImgBBoxAnno);

} // namespace bgm