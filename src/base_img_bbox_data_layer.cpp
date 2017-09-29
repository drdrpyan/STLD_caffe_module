#include "base_img_bbox_data_layer.hpp"

#include <opencv2/core.hpp>

#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"


namespace caffe
{

template <typename Dtype>
BaseImgBBoxDataLayer<Dtype>::BaseImgBBoxDataLayer(const LayerParameter& param)
  : DataLayer<Dtype>(param),
  use_pad_(param.has_padding_param()) {

  if (use_pad_) {
    const PaddingParameter& padding_param = param.padding_param();
    SetPad(padding_param);
  }
}

template <typename Dtype>
void BaseImgBBoxDataLayer<Dtype>::DataLayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Read a data point, and use it to initialize the top blob.
  ImgBBoxAnnoDatum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> data_shape = this->data_transformer_->InferBlobShape(datum.img_datum());
  if (use_pad_) {
    data_shape[2] += (pad_up_ + pad_down_);
    data_shape[3] += (pad_left_ + pad_right_);
  }
  this->transformed_data_.Reshape(data_shape);

  // Reshape prefetch buffers for image data.
  data_shape[0] =layer_param_.data_param().batch_size();
  top[0]->Reshape(data_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(data_shape);
  }

  LOG_IF(INFO, Caffe::root_solver())
    << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();

  if (this->output_labels_) {
    vector<int> gt_shape(4, 1);
    gt_shape[0] = layer_param_.data_param().batch_size();
    gt_shape[1] = 5;

    top[1]->Reshape(gt_shape);

    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(gt_shape);
    }
  }
}

template <typename Dtype>
void BaseImgBBoxDataLayer<Dtype>::ParseLabelBBox(
  const Blob<Dtype>& prefetch_label,
  std::vector<std::vector<Dtype> >* label,
  std::vector<std::vector<bgm::BBox<Dtype> > >* bbox) const {
  CHECK_EQ(prefetch_label.shape()[1], 5);
  CHECK_EQ(prefetch_label.shape()[2], 1);
  CHECK(label);
  CHECK(bbox);

  label->resize(prefetch_label.num());
  bbox->resize(prefetch_label.num());

  for (int n = 0; n < prefetch_label.num(); ++n) {
    const Dtype* label_iter = prefetch_label.cpu_data() + prefetch_label.offset(n, LABEL);
    const Dtype* xmin_iter = prefetch_label.cpu_data() + prefetch_label.offset(n, XMIN);
    const Dtype* ymin_iter = prefetch_label.cpu_data() + prefetch_label.offset(n, YMIN);
    const Dtype* xmax_iter = prefetch_label.cpu_data() + prefetch_label.offset(n, XMAX);
    const Dtype* ymax_iter = prefetch_label.cpu_data() + prefetch_label.offset(n, YMAX);

    (*label)[n].clear();
    (*bbox)[n].clear();

    for (int i = 0; i < prefetch_label.width() && label_iter[i] != LabelParameter::DUMMY_LABEL; ++i) {
      (*label)[n].push_back(label_iter[i]);
      (*bbox)[n].push_back(bgm::BBox<Dtype>(xmin_iter[i], ymin_iter[i], xmax_iter[i], ymax_iter[i]));
    }
  }
}

template <typename Dtype>
void BaseImgBBoxDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  // image data
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());

  if (output_labels_) {
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseImgBBoxDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  // image data
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());

  if (output_labels_) {
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  }
}

template <typename Dtype>
void BaseImgBBoxDataLayer<Dtype>::load_batch(caffe::Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());

  const int& BATCH_SIZE = layer_param_.data_param().batch_size();

  // read datum
  std::vector<ImgBBoxAnnoDatum> img_bbox_anno_datum(BATCH_SIZE);
  for (int i = 0; i < BATCH_SIZE; ++i) {
    timer.Start();

    while (Skip())
      Next();

    img_bbox_anno_datum[i].ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    Next();
  }

  // copy datum
  PrepareBatch(img_bbox_anno_datum, batch);
  for (int i = 0; i < BATCH_SIZE; ++i) {
    timer.Start();
    CopyImage(i, img_bbox_anno_datum[i], &(batch->data_));
    trans_time += timer.MicroSeconds();

    if(this->output_labels_)
      CopyLabel(i, img_bbox_anno_datum[i], &(batch->label_));
  }

  timer.Stop();
  batch_timer.Stop();
  //DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void BaseImgBBoxDataLayer<Dtype>::SetPad(const PaddingParameter& pad_param) {
  pad_type_ = pad_param.type();
  pad_up_ = pad_param.pad_up();
  pad_down_ = pad_param.pad_down();
  pad_left_ = pad_param.pad_left();
  pad_right_ = pad_param.pad_right();
}

template <typename Dtype>
void BaseImgBBoxDataLayer<Dtype>::PrepareBatch(
    const std::vector<ImgBBoxAnnoDatum>& datum,  Batch<Dtype>* batch) {
  CHECK_GT(datum.size(), 0);
  CHECK(batch);

  const int& BATCH_SIZE = layer_param_.data_param().batch_size();

  std::vector<int> data_shape = data_transformer_->InferBlobShape(datum[0].img_datum());
  if (use_pad_) {
    data_shape[2] += (pad_up_ + pad_down_);
    data_shape[3] += (pad_left_ + pad_right_);
  }
  transformed_data_.Reshape(data_shape);

  data_shape[0] = BATCH_SIZE;
  batch->data_.Reshape(data_shape);

  if (output_labels_) {
    int max_num_gt = 0;
    for (int i = 0; i < datum.size(); ++i) {
      int num_gt = datum[i].label().size();
      if (num_gt > max_num_gt)
        max_num_gt = num_gt;
    }

    std::vector<int> label_shape(4);
    label_shape[0] = BATCH_SIZE;
    label_shape[1] = 5; // label, x_min, y_min, x_max, y_max
    label_shape[2] = 1;
    label_shape[3] = max_num_gt;

    batch->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void BaseImgBBoxDataLayer<Dtype>::CopyImage(int item_id, 
                                            const ImgBBoxAnnoDatum& datum,
                                            Blob<Dtype>* batch_data) {
  CHECK(batch_data);
  CHECK_GE(item_id, 0);
  CHECK_LT(item_id, batch_data->num());

  std::vector<int> data_shape = data_transformer_->InferBlobShape(datum.img_datum());
  if (use_pad_) {
    data_shape[2] += (pad_up_ + pad_down_);
    data_shape[3] += (pad_left_ + pad_right_);
  }
  CHECK_EQ(data_shape[1], batch_data->shape()[1]);
  CHECK_EQ(data_shape[2], batch_data->shape()[2]);
  CHECK_EQ(data_shape[3], batch_data->shape()[3]);

  Datum img_datum;
  if (use_pad_) {
    cv::Mat img_mat = DecodeDatumToCVMatNative(datum.img_datum());
    switch (pad_type_) {
      case PaddingParameter::ZERO:
        cv::copyMakeBorder(img_mat, img_mat, pad_up_, pad_down_, pad_left_, pad_right_, IPL_BORDER_CONSTANT);
        break;
      case PaddingParameter::MIRROR:
        cv::copyMakeBorder(img_mat, img_mat, pad_up_, pad_down_, pad_left_, pad_right_, IPL_BORDER_REPLICATE);
        break;
      default:
        LOG(FATAL) << "Illegal padding type";
    }

    CVMatToDatum(img_mat, &img_datum);
  }
  else {
    img_datum.CopyFrom(datum.img_datum());
    DecodeDatumNative(&img_datum);
  }

  Dtype* dst = batch_data->mutable_cpu_data() + batch_data->offset(item_id);
  transformed_data_.set_cpu_data(dst);
  data_transformer_->Transform(img_datum, &transformed_data_);
}

template <typename Dtype>
void BaseImgBBoxDataLayer<Dtype>::CopyLabel(int item_id,
                                            const ImgBBoxAnnoDatum& datum,
                                            Blob<Dtype>* batch_label) {
  CHECK(batch_label);
  CHECK_GE(item_id, 0);
  CHECK_LT(item_id, batch_label->num());
  CHECK_EQ(batch_label->shape()[1], 5);
  CHECK_EQ(batch_label->shape()[2], 1);

  const int NUM_GT = datum.label().size();
  CHECK_LE(NUM_GT, batch_label->width());
  CHECK_EQ(NUM_GT, datum.x_max().size());
  CHECK_EQ(NUM_GT, datum.y_min().size());
  CHECK_EQ(NUM_GT, datum.y_max().size());

  Dtype* label_dst = batch_label->mutable_cpu_data() + batch_label->offset(item_id, LABEL);
  Dtype* xmin_dst = batch_label->mutable_cpu_data() + batch_label->offset(item_id, XMIN);
  Dtype* ymin_dst = batch_label->mutable_cpu_data() + batch_label->offset(item_id, YMIN);
  Dtype* xmax_dst = batch_label->mutable_cpu_data() + batch_label->offset(item_id, XMAX);
  Dtype* ymax_dst = batch_label->mutable_cpu_data() + batch_label->offset(item_id, YMAX);

  for (int i = 0; i < NUM_GT; ++i) {
    label_dst[i] = static_cast<Dtype>(datum.label(i));
    xmin_dst[i] = static_cast<Dtype>(datum.x_min(i));
    ymin_dst[i] = static_cast<Dtype>(datum.y_min(i));
    xmax_dst[i] = static_cast<Dtype>(datum.x_max(i));
    ymax_dst[i] = static_cast<Dtype>(datum.y_max(i));

    if (use_pad_) {
      xmin_dst[i] += pad_left_;
      ymin_dst[i] += pad_up_;
      xmax_dst[i] += pad_left_;
      ymax_dst[i] += pad_up_;
    }
  }
  for (int i = NUM_GT; i < batch_label->width(); ++i) {
    label_dst[i] = static_cast<Dtype>(LabelParameter::DUMMY_LABEL);
    xmin_dst[i] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
    ymin_dst[i] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
    xmax_dst[i] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
    ymax_dst[i] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
  }
}

#ifdef CPU_ONLY
STUB_GPU(BaseImgBBoxDataLayer);
#endif

INSTANTIATE_CLASS(BaseImgBBoxDataLayer);
REGISTER_LAYER_CLASS(BaseImgBBoxData);

} // namespace caffe