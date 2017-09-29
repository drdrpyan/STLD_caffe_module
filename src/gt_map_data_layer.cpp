#include "gt_map_data_layer.hpp"

#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

#include <opencv2/imgproc.hpp>

namespace caffe
{

template <typename Dtype>
GTMapDataLayer<Dtype>::GTMapDataLayer(const LayerParameter& param) 
  : DataLayer<Dtype>(param), BATCH_SIZE_(param.data_param().batch_size()),
    RECEPTIVE_FIELD_WIDTH_(param.gt_map_data_param().receptive_field().width()),
    RECEPTIVE_FIELD_HEIGHT_(param.gt_map_data_param().receptive_field().height()),
    HORIZONTAL_STRIDE_(param.gt_map_data_param().horizontal_stride()),
    VERTICAL_STRIDE_(param.gt_map_data_param().vertical_stride()),
    PATCH_OFFSET_NORMALIZATION_(param.gt_map_data_param().patch_offset_normalization()),
    BBOX_NORMALIZATION_(param.gt_map_data_param().bbox_normalization()),
    USE_PAD_(param.has_padding_param()) {

  if (param.gt_map_data_param().has_activation_region_param()) {
    const ActivationRegionParameter& ar_param = param.gt_map_data_param().activation_region_param();
    activation_method_ = ar_param.method();
    if (ar_param.has_region()) {
      int xmin = ar_param.region().top_left().x();
      int ymin = ar_param.region().top_left().y();
      int xmax = xmin + ar_param.region().size().width() - 1;
      int ymax = ymin + ar_param.region().size().height() - 1;
      activation_region_.Set(xmin, ymin, xmax, ymax);
    }
    else
      activation_region_.Set(0, 0, RECEPTIVE_FIELD_WIDTH_ - 1, RECEPTIVE_FIELD_HEIGHT_ - 1);
  }
  else {
    activation_method_ = ActivationRegionParameter::WHOLE;
    activation_region_.Set(0, 0, RECEPTIVE_FIELD_WIDTH_ - 1, RECEPTIVE_FIELD_HEIGHT_ - 1);
  }

  if (USE_PAD_) {
    const PaddingParameter& padding_param = param.padding_param();
    pad_type_ = padding_param.type();
    pad_up_ = padding_param.pad_up();
    pad_down_ = padding_param.pad_down();
    pad_left_ = padding_param.pad_left();
    pad_right_ = padding_param.pad_right();
  }
}

template <typename Dtype>
void GTMapDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Read a data point, and use it to initialize the top blob.
  ImgBBoxAnnoDatum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> data_shape = this->data_transformer_->InferBlobShape(datum.img_datum());
  if (USE_PAD_) {
    data_shape[2] += (pad_up_ + pad_down_);
    data_shape[3] += (pad_left_ + pad_right_);
  }
  this->transformed_data_.Reshape(data_shape);

  // Reshape prefetch buffers for image data.
  data_shape[0] = BATCH_SIZE_;
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
    gt_shape[0] = BATCH_SIZE_;
    gt_shape[1] = 5;
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(gt_shape);
    }

    int map_width, map_height;
    ComputeMapSize(data_shape[3], data_shape[2], &map_width, &map_height);
    std::vector<int> gt_map_shape(4);
    gt_map_shape[0] = BATCH_SIZE_;
    gt_map_shape[2] = map_height;
    gt_map_shape[3] = map_width;

    // label
    if (top.size() > 1) {
      gt_map_shape[1] = 1;
      label_map_.Reshape(gt_map_shape);
      top[1]->Reshape(gt_map_shape);
      LOG_IF(INFO, Caffe::root_solver())
        << "output label size: " << top[1]->num() << ","
        << top[1]->channels() << "," << top[1]->height() << ","
        << top[1]->width();
    }
    // patch offset
    if (top.size() > 2) {
      gt_map_shape[1] = 4;
      offset_map_.Reshape(gt_map_shape);
      top[2]->Reshape(gt_map_shape);
      LOG_IF(INFO, Caffe::root_solver())
        << "output patch_offset size: " << top[2]->num() << ","
        << top[2]->channels() << "," << top[2]->height() << ","
        << top[2]->width();
    }
    // bbox
    if (top.size() > 3) {
      gt_map_shape[1] = 4;
      bbox_map_.Reshape(gt_map_shape);
      top[3]->Reshape(gt_map_shape);
      LOG_IF(INFO, Caffe::root_solver())
        << "output bbox size: " << top[3]->num() << ","
        << top[3]->channels() << "," << top[3]->height() << ","
        << top[3]->width();
    }

  }
}

template <typename Dtype>
void GTMapDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  // image data
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());


  if (output_labels_) {
    MakeLabelBBoxMap(top[0]->width(), top[0]->height(), prefetch_current_->label_);

    // label
    if (top.size() > 1) {
      top[1]->ReshapeLike(label_map_);
      top[1]->set_cpu_data(label_map_.mutable_cpu_data());
    }

    // patch offset
    if (top.size() > 2) {
      MakeOffsetMap(top[0]->width(), top[0]->height());
      top[2]->ReshapeLike(offset_map_);
      top[2]->set_cpu_data(offset_map_.mutable_cpu_data());
    }

    // bbox
    if (top.size() > 3) {
      top[3]->ReshapeLike(bbox_map_);
      top[3]->set_cpu_data(bbox_map_.mutable_cpu_data());
    }
  }
}

template <typename Dtype>
void GTMapDataLayer<Dtype>::load_batch(caffe::Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());

  // read datum
  std::vector<ImgBBoxAnnoDatum> img_bbox_anno_datum(BATCH_SIZE_);
  for (int i = 0; i < BATCH_SIZE_; ++i) {
    timer.Start();

    while (Skip())
      Next();

    img_bbox_anno_datum[i].ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    Next();
  }

  // copy datum
  PrepareBatch(img_bbox_anno_datum, batch);
  for (int i = 0; i < BATCH_SIZE_; ++i) {
    timer.Start();
    CopyImage(i, img_bbox_anno_datum[i], &(batch->data_));
    trans_time += timer.MicroSeconds();

    if(this->output_labels_)
      CopyLabel(i, img_bbox_anno_datum[i], &(batch->label_));
  }

  timer.Stop();
  batch_timer.Stop();
  //DLOG(INFO) << "Phase         : " << (this->layer_param_.phase() == caffe::Phase::TEST) ? "TEST" : "TRAIN";
  //DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void GTMapDataLayer<Dtype>::ComputeMapSize(int data_width, int data_height,
                                           int* width, int* height) const {
  CHECK_GE(data_width, RECEPTIVE_FIELD_WIDTH_);
  CHECK_GE(data_height, RECEPTIVE_FIELD_HEIGHT_);
  CHECK(width);
  CHECK(height);
  
  *width = ((data_width - RECEPTIVE_FIELD_WIDTH_) / HORIZONTAL_STRIDE_) + 1;
  *height = ((data_height - RECEPTIVE_FIELD_HEIGHT_) / VERTICAL_STRIDE_) + 1;  
}

template <typename Dtype>
void GTMapDataLayer<Dtype>::PrepareBatch(const std::vector<ImgBBoxAnnoDatum>& datum, 
                                         Batch<Dtype>* batch) {
  CHECK_GT(datum.size(), 0);
  CHECK(batch);

  std::vector<int> data_shape = data_transformer_->InferBlobShape(datum[0].img_datum());
  if (USE_PAD_) {
    data_shape[2] += (pad_up_ + pad_down_);
    data_shape[3] += (pad_left_ + pad_right_);
  }
  transformed_data_.Reshape(data_shape);

  data_shape[0] = BATCH_SIZE_;
  batch->data_.Reshape(data_shape);

  if (output_labels_) {
    int max_num_gt = 0;
    for (int i = 0; i < datum.size(); ++i) {
      int num_gt = datum[i].label().size();
      if (num_gt > max_num_gt)
        max_num_gt = num_gt;
    }

    std::vector<int> label_shape(4);
    label_shape[0] = BATCH_SIZE_;
    label_shape[1] = 5; // label, x_min, y_min, x_max, y_max
    label_shape[2] = 1;
    label_shape[3] = max_num_gt;

    batch->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void GTMapDataLayer<Dtype>::CopyImage(int item_id, const ImgBBoxAnnoDatum& datum,
                                      Blob<Dtype>* batch_data) {
  CHECK_GE(item_id, 0);
  CHECK(batch_data);

  std::vector<int> data_shape = data_transformer_->InferBlobShape(datum.img_datum());
  if (USE_PAD_) {
    data_shape[2] += (pad_up_ + pad_down_);
    data_shape[3] += (pad_left_ + pad_right_);
  }
  CHECK_EQ(data_shape[1], batch_data->shape()[1]);
  CHECK_EQ(data_shape[2], batch_data->shape()[2]);
  CHECK_EQ(data_shape[3], batch_data->shape()[3]);

  Datum img_datum;
  if (USE_PAD_) {
    cv::Mat img_mat = DecodeDatumToCVMatNative(datum.img_datum());
    switch (pad_type_) {
      case PaddingParameter::ZERO:
        cv::copyMakeBorder(img_mat, img_mat, pad_up_, pad_down_, pad_left_, pad_right_, IPL_BORDER_CONSTANT);
        break;
      case PaddingParameter::MIRROR:
        cv::copyMakeBorder(img_mat, img_mat, pad_up_, pad_down_, pad_left_, pad_right_, IPL_BORDER_REPLICATE);
        break;
      default:
        LOG(ERROR) << "Illegal padding type";
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
void GTMapDataLayer<Dtype>::CopyLabel(int item_id, const ImgBBoxAnnoDatum& datum,
                                      Blob<Dtype>* batch_label) {
  CHECK_GE(item_id, 0);
  CHECK(batch_label);
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
  }
  for (int i = NUM_GT; i < batch_label->width(); ++i) {
    label_dst[i] = static_cast<Dtype>(LabelParameter::DUMMY_LABEL);
    xmin_dst[i] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
    ymin_dst[i] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
    xmax_dst[i] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
    ymax_dst[i] = static_cast<Dtype>(BBoxParameter::DUMMY_VALUE);
  }
  
}

template <typename Dtype>
void GTMapDataLayer<Dtype>::MakeLabelBBoxMap(int img_width, int img_height, 
                                             const Blob<Dtype>& prefetch_label) {
  int map_width, map_height;
  ComputeMapSize(img_width, img_height, &map_width, &map_height);

  std::vector<int> map_shape(4);
  map_shape[0] = prefetch_label.num();
  map_shape[2] = map_height;
  map_shape[3] = map_width;

  map_shape[1] = 1;
  label_map_.Reshape(map_shape);
  map_shape[1] = 4;
  bbox_map_.Reshape(map_shape);

  std::vector<std::vector<Dtype> > gt_label;
  std::vector<std::vector<bgm::BBox<Dtype> > > gt_bbox;
  ParseLabelBBox(prefetch_label, &gt_label, &gt_bbox);

  if (USE_PAD_) {
    for (int i = 0; i < gt_bbox.size(); ++i)
      for (int j = 0; j < gt_bbox[i].size(); ++j)
        gt_bbox[i][j].Shift(pad_left_, pad_up_);
  }

  for (int n = 0; n < prefetch_label.num(); ++n) {
    int offset_x = 0;
    int offset_y = 0;
    Dtype* label_iter = label_map_.mutable_cpu_data() + label_map_.offset(n);
    Dtype* xmin_iter = bbox_map_.mutable_cpu_data() + bbox_map_.offset(n, 0);
    Dtype* ymin_iter = bbox_map_.mutable_cpu_data() + bbox_map_.offset(n, 1);
    Dtype* width_iter = bbox_map_.mutable_cpu_data() + bbox_map_.offset(n, 2);
    Dtype* height_iter = bbox_map_.mutable_cpu_data() + bbox_map_.offset(n, 3);

    for (int h = 0; h < map_height; ++h) {
      for (int w = 0; w < map_width; ++w) {
        int gt_idx = FindActiveGT(offset_x, offset_y, gt_bbox[n]);
        if (gt_idx != -1) {
          *label_iter = gt_label[n][gt_idx];
          *xmin_iter = gt_bbox[n][gt_idx].x_min() - offset_x;
          *ymin_iter = gt_bbox[n][gt_idx].y_min() - offset_y;
          *width_iter = gt_bbox[n][gt_idx].x_max() - gt_bbox[n][gt_idx].x_min() + 1;
          *height_iter = gt_bbox[n][gt_idx].y_max() - gt_bbox[n][gt_idx].y_min() + 1;

          if (BBOX_NORMALIZATION_) {
            *xmin_iter /= RECEPTIVE_FIELD_WIDTH_;
            *ymin_iter /= RECEPTIVE_FIELD_HEIGHT_;
            *width_iter /= RECEPTIVE_FIELD_WIDTH_;
            *height_iter /= RECEPTIVE_FIELD_HEIGHT_;
          }
        }
        else {
          *label_iter = LabelParameter::NONE;
          *xmin_iter = BBoxParameter::DUMMY_VALUE;
          *ymin_iter = BBoxParameter::DUMMY_VALUE;
          *width_iter = BBoxParameter::DUMMY_VALUE;
          *height_iter = BBoxParameter::DUMMY_VALUE;
        }

        offset_x += HORIZONTAL_STRIDE_;
        ++label_iter;
        ++xmin_iter;
        ++ymin_iter;
        ++width_iter;
        ++height_iter;
      }
      offset_x = 0;
      offset_y += VERTICAL_STRIDE_;
    }
  }
}

template <typename Dtype>
void GTMapDataLayer<Dtype>::ParseLabelBBox(const Blob<Dtype>& prefetch_label,
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
int GTMapDataLayer<Dtype>::FindActiveGT(int offset_x, int offset_y,
                                        const std::vector<bgm::BBox<Dtype> >& bbox) const {
  CHECK_GE(offset_x, 0);
  CHECK_GE(offset_y, 0);

  bgm::BBox<Dtype> activation_region = activation_region_;
  activation_region.Shift(offset_x, offset_y);

  int i;
  for (i = 0; i < bbox.size() && !IsActiveGT(activation_region, bbox[i]); ++i);
  return (i == bbox.size() ? -1 : i);
}

template <typename Dtype>
bool GTMapDataLayer<Dtype>::IsActiveGT(const bgm::BBox<Dtype>& activation_region,
                                       const bgm::BBox<Dtype>& bbox) const {
  bool is_active = false;

  switch (activation_method_) {
    case ActivationRegionParameter::WHOLE:
      if (bbox.x_min() >= activation_region.x_min() &&
          bbox.x_max() <= activation_region.x_max() &&
          bbox.y_min() >= activation_region.y_min() &&
          bbox.y_max() <= activation_region.y_max())
        is_active = true;
      break;
    case ActivationRegionParameter::ANY:
      LOG(ERROR) << "Not implemented yet.";
      break;
    case ActivationRegionParameter::CENTER:
    {
      Dtype center_x = (bbox.x_min() + bbox.x_max()) / 2.0;
      Dtype center_y = (bbox.y_min() + bbox.y_max()) / 2.0;
      if (center_x >= activation_region.x_min() &&
          center_x <= activation_region.x_max() &&
          center_y >= activation_region.y_min() &&
          center_y <= activation_region.y_max())
        is_active = true;
      break;
    }
    default:
      LOG(ERROR) << "Illegal activation method.";
  }

  return is_active;
}

template <typename Dtype>
void GTMapDataLayer<Dtype>::MakeOffsetMap(int img_width, int img_height) {
  int map_width, map_height;
  ComputeMapSize(img_width, img_height, &map_width, &map_height);

  if (map_width != offset_map_.width() || map_height != offset_map_.height()) {
    std::vector<int> offset_shape(4);
    offset_shape[0] = BATCH_SIZE_;
    offset_shape[1] = 4;
    offset_shape[2] = map_height;
    offset_shape[3] = map_width;
    
    offset_map_.Reshape(offset_shape);

    const int& HEIGHT = offset_shape[2];
    const int& WIDTH = offset_shape[3];

    // offset x
    Dtype* offset_x_iter = offset_map_.mutable_cpu_data();
    Dtype stride_x = 
        PATCH_OFFSET_NORMALIZATION_ ? HORIZONTAL_STRIDE_ / static_cast<Dtype>(img_width) : HORIZONTAL_STRIDE_;
    Dtype offset_x = 0;
    for (int i = 0; i < WIDTH; i++) {
      offset_x_iter[i] = offset_x;
      offset_x += stride_x;
    }
    for (int i = HEIGHT - 1; i--; ) {
      caffe_copy(WIDTH, offset_x_iter, offset_x_iter + WIDTH);
      offset_x_iter += WIDTH;
    }

    // offset y
    Dtype* offset_y_iter = offset_map_.mutable_cpu_data() + offset_map_.offset(0, 1);
    Dtype stride_y =
        PATCH_OFFSET_NORMALIZATION_ ? VERTICAL_STRIDE_ / static_cast<Dtype>(img_height) : VERTICAL_STRIDE_;
    Dtype offset_y = 0;
    for (int i = HEIGHT; i--; ) {
      caffe_set(WIDTH, offset_y, offset_y_iter);
      offset_y += stride_y;
      offset_y_iter += WIDTH;
    }

    // window width
    Dtype* rf_width_iter = offset_map_.mutable_cpu_data() + offset_map_.offset(0, 2);
    Dtype rf_width = 
        (PATCH_OFFSET_NORMALIZATION_) ? RECEPTIVE_FIELD_WIDTH_ / static_cast<Dtype>(img_width) : RECEPTIVE_FIELD_WIDTH_;
    caffe_set(WIDTH * HEIGHT, rf_width, rf_width_iter);

    // window height
    Dtype* rf_height_iter = offset_map_.mutable_cpu_data() + offset_map_.offset(0, 3);
    Dtype rf_height = 
        (PATCH_OFFSET_NORMALIZATION_) ? RECEPTIVE_FIELD_HEIGHT_ / static_cast<Dtype>(img_height) : RECEPTIVE_FIELD_HEIGHT_;
    caffe_set(WIDTH * HEIGHT, rf_height, rf_height_iter);

    // copy along n axis
    Dtype* item_iter = offset_map_.mutable_cpu_data();
    int item_size = offset_shape[1] * offset_shape[2] * offset_shape[3];
    for (int i = offset_map_.num() - 1; i--;) {
      caffe_copy(item_size, item_iter, item_iter + item_size);
      item_iter += item_size;
    }
    
  }
}

#ifdef CPU_ONLY
STUB_GPU(GTMapDataLayer);
#endif

INSTANTIATE_CLASS(GTMapDataLayer);
REGISTER_LAYER_CLASS(GTMapData);

} // namespace caffe