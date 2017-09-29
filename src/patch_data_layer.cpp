#include "patch_data_layer.hpp"

#include "caffe/layer_factory.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

#include <opencv2/core.hpp>

namespace caffe
{

template <typename Dtype>
PatchDataLayer<Dtype>::PatchDataLayer(const LayerParameter& param) 
  : DataLayer<Dtype>(param), 
    BATCH_SIZE_(param.data_param().batch_size()),
    NUM_LABEL_(param.label_param().num_label()),
    POSITIVE_ONLY_(param.patch_data_param().positive_only()),
    PATCH_OFFSET_NORMALIZATION_(param.patch_data_param().patch_offset_normalization()),
    BBOX_NORMALIZATION_(param.patch_data_param().bbox_normalization()) {
  CHECK(BATCH_SIZE_ > 0);
}

template <typename Dtype>
void PatchDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Read a data point, and use it to initialize the top blob.
  PatchDatum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = 
      this->data_transformer_->InferBlobShape(datum.patch_img());
  this->transformed_data_.Reshape(top_shape);

  top_shape[0] = BATCH_SIZE_;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }

  LOG_IF(INFO, Caffe::root_solver())
    << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();

  // reshape top[1] (for labels)
  if (this->output_labels_) {
    vector<int> out_shape(4, 1);
    out_shape[0] = BATCH_SIZE_;
    out_shape[1] = 9;
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(out_shape);
    }

    // label
    if (top.size() > 1) {
      out_shape[1] = 1;
      label_.Reshape(out_shape);
      top[1]->Reshape(out_shape);
      LOG_IF(INFO, Caffe::root_solver())
        << "output label size: " << top[1]->num() << ","
        << top[1]->channels() << "," << top[1]->height() << ","
        << top[1]->width();
    }
    // patch offset
    if (top.size() > 2) {
      out_shape[1] = 4;
      patch_offset_.Reshape(out_shape);
      top[2]->Reshape(out_shape);
      LOG_IF(INFO, Caffe::root_solver())
        << "output patch_offset size: " << top[2]->num() << ","
        << top[2]->channels() << "," << top[2]->height() << ","
        << top[2]->width();
    }
    // bbox
    if (top.size() > 3) {
      out_shape[1] = 4;
      bbox_.Reshape(out_shape);
      top[3]->Reshape(out_shape);
      LOG_IF(INFO, Caffe::root_solver())
        << "output bbox size: " << top[3]->num() << ","
        << top[3]->channels() << "," << top[3]->height() << ","
        << top[3]->width();
    }

  }
}

template <typename Dtype>
void PatchDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  // image data
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());

  // label
  if (top.size() > 1) {
    ExtractLabelOut(prefetch_current_->label_, &label_);
    top[1]->ReshapeLike(label_);
    top[1]->set_cpu_data(label_.mutable_cpu_data());
  }

  // patch offset
  if (top.size() > 2) {
    ExtractPatchOffsetOut(prefetch_current_->label_, &patch_offset_);
    top[2]->ReshapeLike(patch_offset_);
    top[2]->set_cpu_data(patch_offset_.mutable_cpu_data());
  }

  // bbox
  if (top.size() > 3) {
    ExtractBBoxOut(prefetch_current_->label_, &bbox_);
    top[3]->ReshapeLike(bbox_);
    top[3]->set_cpu_data(bbox_.mutable_cpu_data());
  }
}

template <typename Dtype>
void PatchDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());

  PatchDatum patch_datum;
  int item_id = 0;
  while(item_id < BATCH_SIZE_) {
    timer.Start();

    // read datum
    while (Skip())
      Next();
    patch_datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    if (POSITIVE_ONLY_ &&
        patch_datum.label() == LabelParameter::NONE) {
      Next();
      continue;
    }
    
    if (item_id == 0)
      PrepareCopy(patch_datum, batch);
    
    // copy datum
    timer.Start();
    CopyImage(item_id, patch_datum, &(batch->data_));
    trans_time += timer.MicroSeconds();

    if(this->output_labels_)
      CopyLabel(item_id, patch_datum, &(batch->label_));

    Next();
    item_id++;
  }

  timer.Stop();
  batch_timer.Stop();
  //DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void PatchDataLayer<Dtype>::PrepareCopy(
    const PatchDatum& datum,
    caffe::Batch<Dtype>* batch) {
  CHECK(batch);

  std::vector<int> top_shape = 
      this->data_transformer_->InferBlobShape(datum.patch_img());
  this->transformed_data_.Reshape(top_shape);

  top_shape[0] = BATCH_SIZE_;
  batch->data_.Reshape(top_shape);

  if (this->output_labels_) {
    std::vector<int> batch_label_shape(4);
    batch_label_shape[0] = BATCH_SIZE_;
    batch_label_shape[1] = 1;
    batch_label_shape[2] = 1;
    batch_label_shape[3] = 9; // label (1), patch offset (4), bbox (4)
    batch->label_.Reshape(batch_label_shape);
  }
}

template <typename Dtype>
void PatchDataLayer<Dtype>::CopyImage(
    int item_id, const PatchDatum& datum, Blob<Dtype>* batch_data) {
  CHECK(batch_data);

  Dtype* data_ptr = batch_data->mutable_cpu_data();
  const int OFFSET = batch_data->offset(item_id);

  Datum decoded_datum;
  decoded_datum.CopyFrom(datum.patch_img());
  DecodeDatumNative(&decoded_datum);

  //cv::Mat debug = DecodeDatumToCVMatNative(datum.patch_img());

  this->transformed_data_.set_cpu_data(data_ptr + OFFSET);
  this->data_transformer_->Transform(
      decoded_datum, &(this->transformed_data_));
}

template <typename Dtype>
void PatchDataLayer<Dtype>::CopyLabel(int item_id,
                                      const PatchDatum& datum,
                                      Blob<Dtype>* batch_label) const {
  CHECK(batch_label);
  CHECK_GE(item_id, 0);
  CHECK(datum.label() != caffe::LabelParameter::DUMMY_LABEL);
  CHECK_LE(datum.label(), NUM_LABEL_);

  const int OFFSET = batch_label->offset(item_id);
  Dtype *label_itr = batch_label->mutable_cpu_data() + OFFSET;

  label_itr[0] = static_cast<Dtype>(datum.label());

  // patch offset
  GetPatchOffset(datum, &(label_itr[1]));
  //label_itr += 4;

  // bbox
  GetBBox(datum, &(label_itr[5]));
  //label_itr += 4;
}

template <typename Dtype>
void PatchDataLayer<Dtype>::GetPatchOffset(const PatchDatum& datum,
                                           Dtype* dst) const {
  if (datum.label() == caffe::LabelParameter::DUMMY_LABEL) {
    dst[0] = static_cast<Dtype>(-1);
    dst[1] = static_cast<Dtype>(-1);
    dst[2] = static_cast<Dtype>(-1);
    dst[3] = static_cast<Dtype>(-1);
  }
  else {
    double xmin = datum.patch_offset_xmin();
    double ymin = datum.patch_offset_ymin();
    double width = transformed_data_.width();
    double height = transformed_data_.height();
    if (PATCH_OFFSET_NORMALIZATION_) {
      xmin /= datum.whole_img_width();
      ymin /= datum.whole_img_height();
      width /= datum.whole_img_width();
      height /= datum.whole_img_height();
    }
    dst[0] = static_cast<Dtype>(xmin);
    dst[1] = static_cast<Dtype>(ymin);
    dst[2] = static_cast<Dtype>(width);
    dst[3] = static_cast<Dtype>(height);
  }
}

template <typename Dtype>
void PatchDataLayer<Dtype>::GetBBox(const PatchDatum& datum,
                                    Dtype* dst) const {
  if (datum.label() == caffe::LabelParameter::NONE) {
    dst[0] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
    dst[1] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
    dst[2] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
    dst[3] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
  }
  else {
    double xmin = datum.bbox_xmin() - datum.patch_offset_xmin();
    double ymin = datum.bbox_ymin() - datum.patch_offset_ymin();
    double width = datum.bbox_xmax() - datum.bbox_xmin() + 1;
    double height = datum.bbox_ymax() - datum.bbox_ymin() + 1;
    if (BBOX_NORMALIZATION_) {
      //xmin /= datum.whole_img_width();
      //ymin /= datum.whole_img_height();
      xmin /= transformed_data_.width();
      ymin /= transformed_data_.height();
      //width = std::log(width / transformed_data_.width());
      //height = std::log(height / transformed_data_.height());
      width /= transformed_data_.width();
      height /= transformed_data_.height();
    }
    dst[0] = static_cast<Dtype>(xmin);
    dst[1] = static_cast<Dtype>(ymin);
    dst[2] = static_cast<Dtype>(width);
    dst[3] = static_cast<Dtype>(height);
  }
}

template <typename Dtype>
void PatchDataLayer<Dtype>::ExtractLabelOut(
    const Blob<Dtype>& prefetched_label,
    Blob<Dtype>* label) const {
  CHECK(label);

  //std::vector<int> label_shape(4, 1);
  //label_shape[0] = BATCH_SIZE_;
  //label->Reshape(label_shape);

  const Dtype* src_iter = prefetched_label.cpu_data();
  Dtype* dst_iter = label->mutable_cpu_data();    
  for (int i = 0; i < BATCH_SIZE_; i++) {
    *dst_iter++ = *src_iter;
    src_iter += 9;
  }
}

template <typename Dtype>
void PatchDataLayer<Dtype>::ExtractPatchOffsetOut(
    const Blob<Dtype>& prefetched_label,
    Blob<Dtype>* patch_offset) const {
  CHECK(patch_offset);

  //std::vector<int> patch_offset_shape(4, 1);
  //patch_offset_shape[0] = BATCH_SIZE_;
  //patch_offset_shape[1] = 4;
  //patch_offset->Reshape(patch_offset_shape);

  const Dtype* src_iter = prefetched_label.cpu_data() + 1;
  Dtype* dst_iter = patch_offset->mutable_cpu_data();    
  for (int i = 0; i < BATCH_SIZE_; i++) {
    *dst_iter++ = src_iter[0];
    *dst_iter++ = src_iter[1];
    *dst_iter++ = src_iter[2];
    *dst_iter++ = src_iter[3];
    src_iter += 9;
  }
}

template <typename Dtype>
void PatchDataLayer<Dtype>::ExtractBBoxOut(
    const Blob<Dtype>& prefetched_label, Blob<Dtype>* bbox) const {
  CHECK(bbox);

  //std::vector<int> bbox_shape(4, 1);
  //bbox_shape[0] = BATCH_SIZE_;
  //bbox_shape[1] = 4;
  //bbox->Reshape(bbox_shape);

  const Dtype* src_iter = prefetched_label.cpu_data() + 5;
  Dtype* dst_iter = bbox->mutable_cpu_data();    
  for (int i = 0; i < BATCH_SIZE_; i++) {
    *dst_iter++ = src_iter[0];
    *dst_iter++ = src_iter[1];
    *dst_iter++ = src_iter[2];
    *dst_iter++ = src_iter[3];
    src_iter += 9;
  }
}

INSTANTIATE_CLASS(PatchDataLayer);
REGISTER_LAYER_CLASS(PatchData);

} // namespace caffe