#include "gt_submap_data_layer.hpp"

#include "bbox.hpp"

#include "caffe/util/rng.hpp"

#include <random>

#include <opencv2/core.hpp>

namespace caffe
{

template <typename Dtype>
GTSubmapDataLayer<Dtype>::GTSubmapDataLayer(const LayerParameter& param) 
  : BaseImgBBoxDataLayer<Dtype>(param),
    SUBMAP_BATCH_SIZE_(param.gt_submap_data_param().submap_batch_size()),
    RECEPTIVE_FIELD_WIDTH_(param.gt_submap_data_param().receptive_field().width()),
    RECEPTIVE_FIELD_HEIGHT_(param.gt_submap_data_param().receptive_field().height()),
    HORIZONTAL_STRIDE_(param.gt_submap_data_param().horizontal_stride()),
    VERTICAL_STRIDE_(param.gt_submap_data_param().vertical_stride()),
    NUM_JITTER_(param.gt_submap_data_param().num_jitter()),
    BBOX_NORMALIZATION_(param.gt_submap_data_param().bbox_normalization()),
    OFFSET_ORIGIN_(param.gt_submap_data_param().offset_param().origin()),
    OFFSET_ANCHOR_(param.gt_submap_data_param().offset_param().anchor()),
    OFFSET_NORMALIZATION_(param.gt_submap_data_param().offset_param().normalize()),
    random_engine_(std::random_device()()){

  CHECK_GT(RECEPTIVE_FIELD_WIDTH_, 0);
  CHECK_GT(RECEPTIVE_FIELD_HEIGHT_, 0);
  CHECK_GE(HORIZONTAL_STRIDE_, 0);
  CHECK_GE(VERTICAL_STRIDE_, 0);

  CHECK_GE(NUM_JITTER_, 0);
  
  if (param.gt_submap_data_param().has_activation_region_param()) {
    const ActivationRegionParameter& ar_param = param.gt_submap_data_param().activation_region_param();
    activation_method_ = ar_param.method();
    if (ar_param.has_region()) {      
      int xmin = ar_param.region().top_left().x();
      int ymin = ar_param.region().top_left().y();
      int xmax = xmin + ar_param.region().size().width() - 1;
      int ymax = ymin + ar_param.region().size().height() - 1;
      
      CHECK_LT(xmax, RECEPTIVE_FIELD_WIDTH_);
      CHECK_LT(ymax, RECEPTIVE_FIELD_HEIGHT_);

      activation_region_.Set(xmin, ymin, xmax, ymax);
    }
    else
      activation_region_.Set(0, 0, RECEPTIVE_FIELD_WIDTH_ - 1, RECEPTIVE_FIELD_HEIGHT_ - 1);
  }
  else {
    activation_method_ = ActivationRegionParameter::WHOLE;
    activation_region_.Set(0, 0, RECEPTIVE_FIELD_WIDTH_ - 1, RECEPTIVE_FIELD_HEIGHT_ - 1);
  }

  InitBaseOffsetMap();
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseImgBBoxDataLayer<Dtype>::DataLayerSetUp(bottom, top);

  // data
  std::vector<int> data_shape(4);
  data_shape[0] = SUBMAP_BATCH_SIZE_;
  data_shape[1] = transformed_data_.channels();
  data_shape[2] = RECEPTIVE_FIELD_HEIGHT_ + (VERTICAL_STRIDE_ * (SUBMAP_HEIGHT_ - 1));
  data_shape[3] = RECEPTIVE_FIELD_WIDTH_ + (HORIZONTAL_STRIDE_ * (SUBMAP_WIDTH_ - 1));
  top[0]->Reshape(data_shape);

  std::vector<int> gt_shape(4);
  gt_shape[0] = SUBMAP_BATCH_SIZE_;
  gt_shape[2] = SUBMAP_HEIGHT_;
  gt_shape[3] = SUBMAP_WIDTH_;
  if (this->output_labels_) {
    // label map
    if (top.size() > 1) {
      gt_shape[1] = 1;
      top[1]->Reshape(gt_shape);
    }
    // bbox map
    if (top.size() > 2) {
      gt_shape[1] = 4;
      top[2]->Reshape(gt_shape);
    }
    // offset map
    if (top.size() > 3) {
      gt_shape[1] = 4;
      top[3]->Reshape(gt_shape);
    }
  }
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  while(top_queue_.size() < SUBMAP_BATCH_SIZE_)
    ExtractSubmap();

  for (int i = 0; i < SUBMAP_BATCH_SIZE_; ++i) {
    CopyTop(i, *(top_queue_.front()), top);
    //top_queue_.pop_back();
    top_queue_.pop_front();

    // debug
    //const Dtype* labels = top[1]->cpu_data();
    //int count = 0;
    //for (int i = 0; i < SUBMAP_BATCH_SIZE_ * 9; ++i)
    //  if (*labels++ != 0)
    //    count++;
    //DLOG(INFO) << "# of postive label : " << count;
  }

}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::InitBaseOffsetMap() {
  std::vector<int> map_shape(4);
  map_shape[0] = 1;
  map_shape[1] = 4;
  map_shape[2] = 3;
  map_shape[3] = 3;

  base_offset_map_.Reshape(map_shape);

  // x
  Dtype offset_row_x[3];
  offset_row_x[0] = 0;
  offset_row_x[1] = HORIZONTAL_STRIDE_;
  offset_row_x[2] = HORIZONTAL_STRIDE_ * 2;
  for (int i = 0; i < 3; ++i) {
    Dtype* x_dst = base_offset_map_.mutable_cpu_data() + 
        base_offset_map_.offset(0, 0, i);
    caffe_copy(3, offset_row_x, x_dst);
  }

  // y
  for (int i = 0; i < 3; ++i) {
    Dtype* y_dst = base_offset_map_.mutable_cpu_data() +
        base_offset_map_.offset(0, 1, i);
    caffe_set(3, static_cast<Dtype>(VERTICAL_STRIDE_*i), y_dst);
  }

  // width
  Dtype* w_dst = base_offset_map_.mutable_cpu_data() +
      base_offset_map_.offset(0, 2);
  caffe_set(9, static_cast<Dtype>(RECEPTIVE_FIELD_WIDTH_), w_dst);

  // height
  Dtype* h_dst = base_offset_map_.mutable_cpu_data() +
      base_offset_map_.offset(0, 3);
  caffe_set(9, static_cast<Dtype>(RECEPTIVE_FIELD_HEIGHT_), h_dst);
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::ExtractSubmap() {
  if (prefetch_current_)
    prefetch_free_.push(prefetch_current_);
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  std::vector<std::vector<Dtype> > gt_label;
  std::vector<std::vector<bgm::BBox<Dtype> > > gt_bbox;
  BaseImgBBoxDataLayer<Dtype>::ParseLabelBBox(prefetch_current_->label_, 
                                              &gt_label, &gt_bbox);

  int img_width = prefetch_current_->data_.width();
  int img_height = prefetch_current_->data_.height();

  for (int i = 0; i < prefetch_current_->label_.num(); ++i) {
    std::vector<bgm::BBox<int> > pick, temp_pick;

    PickPositive(gt_bbox[i], img_width, img_height, &temp_pick);
    pick.insert(pick.end(), temp_pick.cbegin(), temp_pick.cend());

    temp_pick.clear();
    PickSemiPositive(gt_bbox[i], img_width, img_height, &temp_pick);
    pick.insert(pick.end(), temp_pick.cbegin(), temp_pick.cend());

    temp_pick.clear();
    PickNegative(img_width, img_height, &temp_pick);
    pick.insert(pick.end(), temp_pick.cbegin(), temp_pick.cend());

    for (int j = 0; j < pick.size(); ++j) {
      TopBlob* new_top_blob = new TopBlob;
      MakeTopBlob(i, gt_label[i], gt_bbox[i],
                  prefetch_current_->data_, pick[j], new_top_blob);
      top_queue_.push_back(std::shared_ptr<TopBlob>(new_top_blob));
    }
  }

  //caffe::shuffle(top_queue_.begin(), top_queue_.end());  
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::PickPositive(
    const std::vector<bgm::BBox<Dtype> >& gt_bbox,
    int img_width, int img_height,
    std::vector<bgm::BBox<int> >* roi) const {
  CHECK_GT(img_width, 0);
  CHECK_GT(img_height, 0);
  CHECK(roi);

  roi->clear();

  for (int i = 0; i < gt_bbox.size(); ++i) {
    if (activation_method_ == ActivationRegionParameter::CENTER) {
      int x_min, x_max, y_min, y_max;
      GetCenterActivationPatchRange(gt_bbox[i], img_width, img_height,
                              &x_min, &x_max, &y_min, &y_max);
      x_max = std::max(x_max, x_min);
      y_max = std::max(y_max, y_min);

      int num_candidate = (x_max - x_min + 1)*(y_max - y_min + 1);
      num_candidate = std::min(num_candidate, static_cast<int>(NUM_JITTER_));

      std::vector<int> roi_x, roi_y;
      GetUniformRandom(num_candidate, x_min, x_max, &roi_x);
      GetUniformRandom(num_candidate, y_min, y_max, &roi_y);

      for (int j = 0; j < num_candidate; ++j) {
        bgm::BBox<int> candidate(
            roi_x[j], roi_y[j],
            roi_x[j] + (RECEPTIVE_FIELD_WIDTH_ + HORIZONTAL_STRIDE_ * 2) - 1,
            roi_y[j] + (RECEPTIVE_FIELD_HEIGHT_ + VERTICAL_STRIDE_ * 2) - 1);
        roi->push_back(candidate);
      }
    }
    else
      LOG(FATAL) << "Not implemented yet.";
  }
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::GetCenterActivationPatchRange(
    const bgm::BBox<Dtype>& gt, int img_width, int img_height,
    int* x_min, int* x_max, int* y_min, int* y_max) const {
  CHECK_GT(img_width, 0);
  CHECK_GT(img_height, 0);
  CHECK(x_min);
  CHECK(x_max);
  CHECK(y_min);
  CHECK(y_max);

  const int GT_CENTER_X = (gt.x_max() + gt.x_min()) / 2;
  const int GT_CENTER_Y = (gt.y_max() + gt.y_min()) / 2;
  const int PATCH_WIDTH = RECEPTIVE_FIELD_WIDTH_ + HORIZONTAL_STRIDE_ * 2;
  const int PATCH_HEIGHT = RECEPTIVE_FIELD_HEIGHT_ + VERTICAL_STRIDE_ * 2;

  CHECK_GE(img_width, PATCH_WIDTH);
  CHECK_GE(img_height, PATCH_HEIGHT);

  int roi_xmin = std::max(
      0, static_cast<int>(GT_CENTER_X - activation_region_.x_max() - HORIZONTAL_STRIDE_));
  roi_xmin = std::min(roi_xmin, img_width - PATCH_WIDTH);
  int roi_xmax = std::max(
      0, static_cast<int>(GT_CENTER_X - activation_region_.x_min() - HORIZONTAL_STRIDE_));
  roi_xmax = std::min(roi_xmax, img_width - PATCH_WIDTH);
  int roi_ymin = std::max(
      0, static_cast<int>(GT_CENTER_Y - activation_region_.y_max() - VERTICAL_STRIDE_));
  roi_ymin = std::min(roi_ymin, img_height - PATCH_HEIGHT);
  int roi_ymax = std::max(
      0, static_cast<int>(GT_CENTER_Y - activation_region_.y_min() - VERTICAL_STRIDE_));
  roi_ymax = std::min(roi_ymax, img_height - PATCH_HEIGHT);

  *x_min = roi_xmin;
  *x_max = roi_xmax;
  *y_min = roi_ymin;
  *y_max = roi_ymax;
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::PickSemiPositive(
    const std::vector<bgm::BBox<Dtype> >& gt_bbox,
    int img_width, int img_height,
    std::vector<bgm::BBox<int> >* roi) const {
  CHECK_GT(img_width, 0);
  CHECK_GT(img_height, 0);
  CHECK(roi);

  roi->clear();

  for (int i = 0; i < gt_bbox.size(); ++i) {
    int x_min, x_max, y_min, y_max;
    GetSemiPositiveRange(gt_bbox[i], img_width, img_height,
                         &x_min, &x_max, &y_min, &y_max);
    x_max = std::max(x_max, x_min);
    y_max = std::max(y_max, y_min);

    int num_candidate = (x_max - x_min + 1)*(y_max - y_min + 1);
    num_candidate = std::min(num_candidate, static_cast<int>(NUM_JITTER_));

    std::vector<int> roi_x, roi_y;
    GetUniformRandom(num_candidate, x_min, x_max, &roi_x);
    GetUniformRandom(num_candidate, y_min, y_max, &roi_y);

    for (int j = 0; j < num_candidate; ++j) {
      bgm::BBox<int> candidate(
          roi_x[j], roi_y[j],
          roi_x[j] + (RECEPTIVE_FIELD_WIDTH_ + HORIZONTAL_STRIDE_ * 2) - 1,
          roi_y[j] + (RECEPTIVE_FIELD_HEIGHT_ + VERTICAL_STRIDE_ * 2) - 1);
      roi->push_back(candidate);
    }
  }
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::GetSemiPositiveRange(
    const bgm::BBox<Dtype>& gt, int img_width, int img_height,
    int* x_min, int* x_max, int* y_min, int* y_max) const {
  CHECK_GT(img_width, 0);
  CHECK_GT(img_height, 0);
  CHECK(x_min);
  CHECK(x_max);
  CHECK(y_min);
  CHECK(y_max);

  const int PATCH_WIDTH = RECEPTIVE_FIELD_WIDTH_ + HORIZONTAL_STRIDE_ * 2;
  const int PATCH_HEIGHT = RECEPTIVE_FIELD_HEIGHT_ + VERTICAL_STRIDE_ * 2;

  CHECK_GE(img_width, PATCH_WIDTH);
  CHECK_GE(img_height, PATCH_HEIGHT);

  *x_min = std::max(0, static_cast<int>(gt.x_max() - PATCH_WIDTH + 1));
  *x_max = std::min(img_width - PATCH_WIDTH, static_cast<int>(gt.x_min()));
  *y_min = std::max(0, static_cast<int>(gt.y_max() - PATCH_HEIGHT + 1));
  *y_max = std::min(img_height - PATCH_HEIGHT, static_cast<int>(gt.y_min()));
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::PickNegative(
    int img_width, int img_height,
    std::vector<bgm::BBox<int> >* roi) const {
  CHECK_GT(img_width, 0);
  CHECK_GT(img_height, 0);
  CHECK(roi);

  roi->clear();

  const int PATCH_WIDTH = RECEPTIVE_FIELD_WIDTH_ + HORIZONTAL_STRIDE_ * 2;
  const int PATCH_HEIGHT = RECEPTIVE_FIELD_HEIGHT_ + VERTICAL_STRIDE_ * 2;

  CHECK_GE(img_width, PATCH_WIDTH);
  CHECK_GE(img_height, PATCH_HEIGHT);

  int x_max = img_width - PATCH_WIDTH;
  int y_max = img_height - PATCH_HEIGHT;

  std::vector<int> x_range, y_range;
  GetUniformRandom(NUM_JITTER_*2, 0, x_max, &x_range);
  GetUniformRandom(NUM_JITTER_*2, 0, y_max, &y_range);

  for (int i = 0; i < x_range.size(); ++i) {
    bgm::BBox<int> neg(x_range[i], y_range[i],
                       x_range[i] + PATCH_WIDTH - 1,
                       y_range[i] + PATCH_WIDTH - 1);
    roi->push_back(neg);
  }

}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::GetUniformRandom(int num, int min, int max, 
                                                std::vector<int>* random) const {
  CHECK_GT(num, 0);
  CHECK_GE(max, min);
  CHECK(random);

  std::uniform_int_distribution<int> dist(min, max);
  auto random_generator = std::bind(dist, random_engine_);

  random->resize(num);
  for (auto iter = random->begin(); iter != random->end(); ++iter)
    *iter = random_generator();
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::MakeTopBlob(
    int data_id,
    const std::vector<Dtype>& gt_label, 
    const std::vector<bgm::BBox<Dtype> >& gt_bbox,
    const Blob<Dtype>& data, const bgm::BBox<int>& roi,
    TopBlob* top_blob) const {
  CHECK(top_blob);

  MakeTopData(data_id, data, roi, &(top_blob->data));
  MakeLabelBBoxMap(gt_label, gt_bbox, roi,
                   &(top_blob->label), &(top_blob->bbox));
  MakeOffsetMap(data.width(), data.height(), roi, &(top_blob->offset));
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::MakeTopData(int data_id,
                                           const Blob<Dtype>& data,
                                           const bgm::BBox<int>& roi,
                                           Blob<Dtype>* top_data) const {
  CHECK_GE(data_id, 0);
  CHECK_LT(data_id, data.num());
  CHECK(top_data);

  const int PATCH_WIDTH = RECEPTIVE_FIELD_WIDTH_ + HORIZONTAL_STRIDE_ * 2;
  const int PATCH_HEIGHT = RECEPTIVE_FIELD_HEIGHT_ + VERTICAL_STRIDE_ * 2;

  CHECK_EQ(PATCH_WIDTH, roi.x_max() - roi.x_min() + 1);
  CHECK_GE(roi.x_min(), 0);
  CHECK_LE(roi.x_max(), data.width() - 1);

  CHECK_EQ(PATCH_HEIGHT, roi.y_max() - roi.y_min() + 1);
  CHECK_GE(roi.y_min(), 0);
  CHECK_LE(roi.y_max(), data.height() - 1);

  std::vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = data.channels();
  top_shape[2] = PATCH_HEIGHT;
  top_shape[3] = PATCH_WIDTH;
  top_data->Reshape(top_shape);

  for (int c = 0; c < data.channels(); ++c) {
    const Dtype* src_iter = data.cpu_data() + 
      data.offset(data_id, c, roi.y_min(), roi.x_min());
    Dtype* dst_iter = top_data->mutable_cpu_data() + top_data->offset(0, c);

    for (int h = 0; h < PATCH_HEIGHT; ++h) {
      caffe_copy(PATCH_WIDTH, src_iter, dst_iter);
      src_iter += data.width();
      dst_iter += PATCH_WIDTH;
    }
  }


  // debug
  //int img_width = top_data->width();
  //int img_height = top_data->height();
  //int img_size = img_width * img_height;
  //Dtype* data_dst = top_data->mutable_cpu_data();
  //std::vector<cv::Mat> bgr(3);
  //bgr[0] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst);
  //bgr[1] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst + img_size);
  //bgr[2] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst + img_size * 2);
  //cv::Mat debug_data(cv::Size(img_width, img_height), CV_32FC3);
  //cv::merge(bgr, debug_data);
  //debug_data.convertTo(debug_data, CV_8UC3);
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::MakeLabelBBoxMap(
    const std::vector<Dtype>& gt_label, 
    const std::vector<bgm::BBox<Dtype> >& gt_bbox,
    const bgm::BBox<int>& roi,
    Blob<Dtype>* label_map, Blob<Dtype>* bbox_map) const {
  CHECK(label_map);
  CHECK(bbox_map);

  std::vector<int> map_shape(4);
  map_shape[0] = 1;
  map_shape[2] = 3;
  map_shape[3] = 3;

  map_shape[1] = 1;
  label_map->Reshape(map_shape);

  map_shape[1] = 4;
  bbox_map->Reshape(map_shape);

  int offset_x = roi.x_min();
  int offset_y = roi.y_min();
  Dtype* label_iter = label_map->mutable_cpu_data();
  Dtype* xmin_iter = bbox_map->mutable_cpu_data();
  Dtype* ymin_iter = bbox_map->mutable_cpu_data() + bbox_map->offset(0, 1);
  Dtype* width_iter = bbox_map->mutable_cpu_data() + bbox_map->offset(0, 2);
  Dtype* height_iter = bbox_map->mutable_cpu_data() + bbox_map->offset(0, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int gt_idx = FindActiveGT(offset_x, offset_y, gt_bbox);
      if (gt_idx != -1) {
        *label_iter = gt_label[gt_idx];
        *xmin_iter = gt_bbox[gt_idx].x_min() - offset_x;
        *ymin_iter = gt_bbox[gt_idx].y_min() - offset_y;
        *width_iter = gt_bbox[gt_idx].x_max() - gt_bbox[gt_idx].x_min() + 1;
        *height_iter = gt_bbox[gt_idx].y_max() - gt_bbox[gt_idx].y_min() + 1;

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

    offset_x = roi.x_min();
    offset_y += VERTICAL_STRIDE_;
  }
}

template <typename Dtype>
int GTSubmapDataLayer<Dtype>::FindActiveGT(
    int offset_x, int offset_y, 
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
bool GTSubmapDataLayer<Dtype>::IsActiveGT(
    const bgm::BBox<Dtype>& activation_region, 
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
      LOG(FATAL) << "Not implemented yet.";
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
      LOG(FATAL) << "Illegal activation method.";
  }

  return is_active;
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::MakeOffsetMap(int img_width, int img_height,
                                             const bgm::BBox<int>& roi,
                                             Blob<Dtype>* offset_map) const {
  CHECK(offset_map);

  offset_map->CopyFrom(base_offset_map_, false, true);

  // x
  Dtype* x_iter = offset_map->mutable_cpu_data() + offset_map->offset(0, 0);
  caffe_add_scalar(9, static_cast<Dtype>(roi.x_min()), x_iter);
  if (OFFSET_NORMALIZATION_)
    caffe_scal<Dtype>(9, static_cast<Dtype>(1.0 / img_width), x_iter);

  // y
  Dtype* y_iter = offset_map->mutable_cpu_data() + offset_map->offset(0, 1);
  caffe_add_scalar(9, static_cast<Dtype>(roi.y_min()), y_iter);
  if (OFFSET_NORMALIZATION_)
    caffe_scal<Dtype>(9, static_cast<Dtype>(1.0 / img_height), y_iter);

  // width
  if (OFFSET_NORMALIZATION_) {
    Dtype* w_iter = offset_map->mutable_cpu_data() + offset_map->offset(0, 2);
    caffe_scal<Dtype>(9, static_cast<Dtype>(1.0 / img_width), w_iter);
  }

  // height
  if (OFFSET_NORMALIZATION_) {
    Dtype* h_iter = offset_map->mutable_cpu_data() + offset_map->offset(0, 3);
    caffe_scal<Dtype>(9, static_cast<Dtype>(1.0 / img_height), h_iter);
  }
}

template <typename Dtype>
void GTSubmapDataLayer<Dtype>::CopyTop(int batch_idx, const TopBlob& top_blob,
                                       const std::vector<Blob<Dtype>*>& top) const {
  CHECK_GE(batch_idx, 0);
  CHECK_LT(batch_idx, SUBMAP_BATCH_SIZE_);
  CHECK_GE(top.size(), 1);

  // data
  cv::Mat debug_data, debug_label;
  cv::Mat debug_bbox_x, debug_bbox_y, debug_bbox_w, debug_bbox_h;
  cv::Mat debug_offset_x, debug_offset_y, debug_offset_w, debug_offset_h;

  const Dtype* data_src = top_blob.data.cpu_data();
  Dtype* data_dst = top[0]->mutable_cpu_data() + top[0]->offset(batch_idx);
  caffe_copy(top_blob.data.count(), data_src, data_dst);
  //// debug
  //int img_width = top_blob.data.width();
  //int img_height = top_blob.data.height();
  //int img_size = img_width * img_height;
  //std::vector<cv::Mat> bgr(3);
  //bgr[0] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst);
  //bgr[1] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst + img_size);
  //bgr[2] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst + img_size * 2);
  //debug_data = cv::Mat(cv::Size(img_width, img_height), CV_32FC3);
  //cv::merge(bgr, debug_data);
  //debug_data.convertTo(debug_data, CV_8UC3);


  // label
  if (top.size() > 1) {
    const Dtype* label_src = top_blob.label.cpu_data();
    Dtype* label_dst = top[1]->mutable_cpu_data() + top[1]->offset(batch_idx);
    caffe_copy(top_blob.label.count(), label_src, label_dst);

    //// debug
    //debug_label = cv::Mat(cv::Size(top_blob.label.width(),
    //                               top_blob.label.height()),
    //                      CV_32FC1, label_dst);
  }

  // bbox
  if (top.size() > 2) {
    const Dtype* bbox_src = top_blob.bbox.cpu_data();
    Dtype* bbox_dst = top[2]->mutable_cpu_data() + top[2]->offset(batch_idx);
    caffe_copy(top_blob.bbox.count(), bbox_src, bbox_dst);

    //// debug
    //debug_bbox_x = cv::Mat(cv::Size(top_blob.bbox.width(),
    //                                top_blob.bbox.height()),
    //                       CV_32FC1, bbox_dst);
    //debug_bbox_y = cv::Mat(cv::Size(top_blob.bbox.width(),
    //                                top_blob.bbox.height()),
    //                       CV_32FC1, bbox_dst + 9);
    //debug_bbox_w = cv::Mat(cv::Size(top_blob.bbox.width(),
    //                                top_blob.bbox.height()),
    //                       CV_32FC1, bbox_dst + 18);
    //debug_bbox_h = cv::Mat(cv::Size(top_blob.bbox.width(),
    //                                top_blob.bbox.height()),
    //                       CV_32FC1, bbox_dst + 27);
  }

  // offset
  if (top.size() > 3) {
    const Dtype* offset_src = top_blob.offset.cpu_data();
    Dtype* offset_dst = top[3]->mutable_cpu_data() + top[3]->offset(batch_idx);
    caffe_copy(top_blob.offset.count(), offset_src, offset_dst);

    //// debug
    //debug_offset_x = cv::Mat(cv::Size(top_blob.bbox.width(),
    //                                top_blob.bbox.height()),
    //                       CV_32FC1, offset_dst);
    //debug_offset_y = cv::Mat(cv::Size(top_blob.bbox.width(),
    //                                top_blob.bbox.height()),
    //                       CV_32FC1, offset_dst + 9);
    //debug_offset_w = cv::Mat(cv::Size(top_blob.bbox.width(),
    //                                top_blob.bbox.height()),
    //                       CV_32FC1, offset_dst + 18);
    //debug_offset_h = cv::Mat(cv::Size(top_blob.bbox.width(),
    //                                top_blob.bbox.height()),
    //                       CV_32FC1, offset_dst + 27);
  }
}

#ifdef CPU_ONLY
STUB_GPU(GTSubmapDataLayer);
#endif

INSTANTIATE_CLASS(GTSubmapDataLayer);
REGISTER_LAYER_CLASS(GTSubmapData);
} // namespace caffe