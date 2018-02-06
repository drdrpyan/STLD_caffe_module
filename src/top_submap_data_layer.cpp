#include "top_submap_data_layer.hpp"

#include "caffe/util/math_functions.hpp"


#ifndef NDEBUG
#include <opencv2/core.hpp>  
#endif // !NDEBUG


namespace caffe
{

template <typename Dtype>
TopSubmapDataLayer<Dtype>::TopSubmapDataLayer(const LayerParameter& param) 
  : BaseImgBBoxDataLayer<Dtype>(param),
    IMG_WIDTH_(param.top_submap_data_param().img_size().width()),
    IMG_HEIGHT_(param.top_submap_data_param().img_size().height()),
    SUBMAP_BATCH_SIZE_(param.top_submap_data_param().submap_batch_size()), 
    WINDOW_WIDTH_(param.top_submap_data_param().win_size().width()),
    WINDOW_HEIGHT_(param.top_submap_data_param().win_size().height()),
    WIN_HORIZONTAL_STRIDE_(param.top_submap_data_param().win_horizontal_stride()),
    WIN_VERTICAL_STRIDE_(param.top_submap_data_param().win_vertical_stride()),
    OFFSET_NORMALIZE_(param.top_submap_data_param().offset_normalize()) {
  const TopSubmapDataParameter& data_param = param.top_submap_data_param();
  
  InitUnitParam(data_param);
  InitNumSubmaps(data_param);
  InitSubmapSize();
  InitOffsetMaps();
  InitBBoxParam(data_param);
}

template <typename Dtype>
void TopSubmapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  BaseImgBBoxDataLayer<Dtype>::DataLayerSetUp(bottom, top);

  // data
  std::vector<int> data_shape(4);
  data_shape[0] = SUBMAP_BATCH_SIZE_;
  data_shape[1] = transformed_data_.channels();
  data_shape[2] = WINDOW_HEIGHT_;
  data_shape[3] = WINDOW_WIDTH_;
  top[0]->Reshape(data_shape);

  // offset
  std::vector<int> offset_shape(4);
  offset_shape[0] = SUBMAP_BATCH_SIZE_;
  offset_shape[1] = 2;
  offset_shape[2] = offset_submap_height_;
  offset_shape[3] = offset_submap_width_;
  top[1]->Reshape(offset_shape);

  std::vector<int> gt_shape(4);
  gt_shape[0] = SUBMAP_BATCH_SIZE_;
  gt_shape[2] = gt_submap_height_;
  gt_shape[3] = gt_submap_width_;

  // label
  if (top.size() > 2) {
    gt_shape[1] = 1;
    top[2]->Reshape(gt_shape);
  }

  // bbox
  if (top.size() > 3) {
    gt_shape[1] = 4;
    top[3]->Reshape(gt_shape);
  }
}

template <typename Dtype>
void TopSubmapDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  while(top_queue_.size() < SUBMAP_BATCH_SIZE_)
    ExtractSubmap();

  for (int i = 0; i < SUBMAP_BATCH_SIZE_; ++i) {
    CopyTop(i, *(top_queue_.front()), top);
    top_queue_.pop_front();
  }
}

template <typename Dtype>
void TopSubmapDataLayer<Dtype>::InitUnitParam(const TopSubmapDataParameter& param) {
  if (param.has_gt_unit()) {
    CHECK(param.has_gt_horizontal_stride());
    CHECK(param.has_gt_vertical_stride());

    gt_unit_width_ = param.gt_unit().width();
    gt_unit_height_ = param.gt_unit().height();
    gt_h_stride_ = param.gt_horizontal_stride();
    gt_v_stride_ = param.gt_vertical_stride();
  }
  else {
    gt_unit_width_ = WINDOW_WIDTH_;
    gt_unit_height_ = WINDOW_HEIGHT_;
    gt_h_stride_ = 1;
    gt_v_stride_ = 1;
  }

  if (param.has_offset_unit()) {
    CHECK(param.has_offset_horizontal_stride());
    CHECK(param.has_offset_vertical_stride());

    offset_unit_width_ = param.offset_unit().width();
    offset_unit_height_ = param.offset_unit().height();
    offset_h_stride_ = param.offset_horizontal_stride();
    offset_v_stride_ = param.offset_vertical_stride();
  }
  else {
    offset_unit_width_ = WINDOW_WIDTH_;
    offset_unit_height_ = WINDOW_HEIGHT_;
    offset_h_stride_ = 1;
    offset_v_stride_ = 1;
  }


  //if (param.has_cell_size()) {
  //  CHECK(param.has_cell_horizontal_stride());
  //  CHECK(param.has_cell_vertical_stride());

  //  cell_width_ = param.cell_size().width();
  //  cell_height_ = param.cell_size().height();
  //  cell_horizontal_stride_ = param.cell_horizontal_stride();
  //  cell_vertical_stride_ = param.cell_vertical_stride();
  //}
  //else {
  //  cell_width_ = WINDOW_WIDTH_;
  //  cell_height_ = WINDOW_HEIGHT_;
  //  cell_horizontal_stride_ = 1;
  //  cell_vertical_stride_ = 1;
  //}
}

template <typename Dtype>
inline void TopSubmapDataLayer<Dtype>::InitNumSubmaps(
    const TopSubmapDataParameter& param) {
  //num_submap_rows_ = (WIN_HORIZONTAL_STRIDE_ > 0) ? 
  //    ((IMG_WIDTH_ - WINDOW_WIDTH_) / WIN_HORIZONTAL_STRIDE_ + 1) : 1;
  //
  //num_submap_cols_ = (WIN_VERTICAL_STRIDE_ > 0) ? 
  //    ((IMG_HEIGHT_ - WINDOW_HEIGHT_) / WIN_VERTICAL_STRIDE_ + 1) : 1;
  if (WIN_HORIZONTAL_STRIDE_ == 0)
    num_submap_rows_ = 1;
  else {
    num_submap_rows_ = (IMG_WIDTH_ - WINDOW_WIDTH_) / WIN_HORIZONTAL_STRIDE_ + 1;
    if (param.has_num_win_rows())
      num_submap_rows_ = std::min(num_submap_rows_,
                                  static_cast<int>(param.num_win_rows()));
  }

  if (WIN_VERTICAL_STRIDE_ == 0)
    num_submap_cols_ = 1;
  else {
    num_submap_cols_ = (IMG_HEIGHT_ - WINDOW_HEIGHT_) / WIN_VERTICAL_STRIDE_ + 1;
    if (param.has_num_win_cols())
      num_submap_cols_ = std::min(num_submap_cols_, 
                                  static_cast<int>(param.num_win_cols()));
  }
}

template <typename Dtype>
void TopSubmapDataLayer<Dtype>::InitOffsetMaps() {
  Blob<Dtype> base;
  InitBaseOffsetMap(&base);
  int map_size = base.width() * base.height();

  // 2차원 벡터에서 1차원으로 바꿀 것
  //offset_map_.resize(num_submap_row_);
  //for (int i = 0; i < num_submap_row_; ++i)
  //  offset_map_[i].resize(num_submap_col_);
  //offset_map_.resize(num_submap_row_ * num_submap_col_);
  MallocOffsetMap();

  float win_topleft_x = 0;
  float win_topleft_y = 0;
  float x_step = WIN_HORIZONTAL_STRIDE_;
  float y_step = WIN_VERTICAL_STRIDE_;
  if (bbox_normalize_) {
    x_step /= IMG_WIDTH_;
    y_step /= IMG_HEIGHT_;
  }

  for (int h = 0; h < num_submap_cols_; ++h) {
    for (int w = 0; w < num_submap_rows_; ++w) {
      //Blob<Dtype>& map = offset_map_[h][w];
      Blob<Dtype>& map = GetOffsetMap(h, w);
      map.CopyFrom(base, false, true);

      Dtype* x_ptr = map.mutable_cpu_data();
      Dtype* y_ptr = map.mutable_cpu_data() + map.offset(0, 1);

      caffe_add_scalar(map_size, static_cast<Dtype>(win_topleft_x), x_ptr);
      caffe_add_scalar(map_size, static_cast<Dtype>(win_topleft_y), y_ptr);

      win_topleft_x += x_step;
    }

    win_topleft_x = 0;
    win_topleft_y += y_step;
  }
}

template <typename Dtype>
void TopSubmapDataLayer<Dtype>::InitBaseOffsetMap(Blob<Dtype>* base) {
  CHECK(base);

  std::vector<int> base_shape(4, 1);
  base_shape[1] = 2;
  base_shape[2] = offset_submap_height_;
  base_shape[3] = offset_submap_width_;
  base->Reshape(base_shape);

  // x
  Dtype h_step = offset_h_stride_;
  if (OFFSET_NORMALIZE_) 
    h_step /= IMG_WIDTH_;
  Dtype* offset_x_1st_row = base->mutable_cpu_data();
  for (int i = 0; i < offset_submap_width_; ++i)
    offset_x_1st_row[i] = h_step * i;
  for (int i = 1; i < offset_submap_height_; ++i) {
    Dtype* x_dst = base->mutable_cpu_data() + base->offset(0, 0, i);
    caffe_copy(offset_submap_width_, offset_x_1st_row, x_dst);
  }

  // y
  Dtype v_step = offset_v_stride_;
  if (OFFSET_NORMALIZE_)
    v_step /= IMG_HEIGHT_;
  for (int i = 0; i < offset_submap_height_; ++i) {
    Dtype* y_dst = base->mutable_cpu_data() + base->offset(0, 1, i);
    caffe_set(offset_submap_width_, static_cast<Dtype>(v_step*i), y_dst);
  }
}

template <typename Dtype>
void TopSubmapDataLayer<Dtype>::InitBBoxParam(
    const TopSubmapDataParameter& param) {
  if (param.has_bbox_param()) {
    if (param.bbox_param().has_anchor())
      bbox_anchor_ = param.bbox_param().anchor();
    else
      bbox_anchor_ = BBoxParameter::TOP_LEFT;

    if (param.bbox_param().has_normalize())
      bbox_normalize_ = param.bbox_param().normalize();
    else
      bbox_normalize_ = true;
  }
}

template <typename Dtype>
const Blob<Dtype>& TopSubmapDataLayer<Dtype>::GetOffsetMap(
    int row, int col) const {
  CHECK_GE(row, 0);
  CHECK_LT(row, num_submap_cols_);
  CHECK_GE(col, 0);
  CHECK_LT(col, num_submap_rows_);
  return *(offset_map_[row * num_submap_rows_ + col]);
}

template <typename Dtype>
void TopSubmapDataLayer<Dtype>::ExtractSubmap() {
  if (prefetch_current_)
    prefetch_free_.push(prefetch_current_);
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  std::vector<std::vector<Dtype> > gt_label;
  std::vector<std::vector<bgm::BBox<Dtype> > > gt_bbox;
  BaseImgBBoxDataLayer<Dtype>::ParseLabelBBox(prefetch_current_->label_, 
                                              &gt_label, &gt_bbox);

  int lefttop_x = 0;
  int lefttop_y = 0;
  for (int i = 0; i < prefetch_current_->label_.num(); ++i) {
    for (int h = 0; h < num_submap_cols_; ++h) {
      for (int w = 0; w < num_submap_rows_; ++w) {
        TopBlob* top_blob = new TopBlob;

        CropImg(prefetch_current_->data_, i,
                lefttop_x, lefttop_y, WINDOW_WIDTH_, WINDOW_HEIGHT_,
                &(top_blob->data));
        
        bool contain_obj = MakeLabelBBoxMap(gt_label[i], gt_bbox[i],
                                            lefttop_x, lefttop_y, 
                                            WINDOW_WIDTH_, WINDOW_HEIGHT_,
                                            &(top_blob->label), 
                                            &(top_blob->bbox));

        //top_blob->offset.CopyFrom(offset_map_[h][w], false, true);
        top_blob->offset.CopyFrom(GetOffsetMap(h, w), false, true);

        if (!contain_obj)
          delete top_blob;
        else
          top_queue_.push_back(std::shared_ptr<TopBlob>(top_blob));

        lefttop_x += WIN_HORIZONTAL_STRIDE_;
      }

      lefttop_x = 0;
      lefttop_y += WIN_VERTICAL_STRIDE_;
    }
  }
}

template <typename Dtype>
void TopSubmapDataLayer<Dtype>::CropImg(const Blob<Dtype>& src, int idx,
                                        int roi_x, int roi_y,
                                        int roi_width, int roi_height,
                                        Blob<Dtype>* dst) const {
  CHECK_GE(roi_x, 0);
  CHECK_GE(roi_y, 0);
  CHECK_GE(roi_width, 0);
  CHECK_GE(roi_height, 0);
  CHECK_LE(roi_x + roi_width, IMG_WIDTH_);
  CHECK_LE(roi_y + roi_height, IMG_HEIGHT_);
  CHECK(dst);

  std::vector<int> dst_shape = src.shape();
  dst_shape[2] = roi_height;
  dst_shape[3] = roi_width;
  dst->Reshape(dst_shape);

  for (int c = 0; c < src.channels(); ++c) {
    for (int h = 0; h < roi_height; ++h) {
      const Dtype* src_ptr = src.cpu_data() + src.offset(idx, c, h + roi_y, roi_x);
      Dtype* dst_ptr = dst->mutable_cpu_data() + dst->offset(0, c, h);
      caffe_copy(roi_width, src_ptr, dst_ptr);
    }
  }
}

template <typename Dtype>
bool TopSubmapDataLayer<Dtype>::MakeLabelBBoxMap(
    const std::vector<Dtype>& gt_label,
    const std::vector<bgm::BBox<Dtype> >& gt_bbox,
    int roi_x, int roi_y, int roi_width, int roi_height,
    Blob<Dtype>* label_map, Blob<Dtype>* bbox_map) const {
  CHECK(label_map);
  CHECK(bbox_map);

  bool contain_object = false;

  std::vector<int> map_shape(4);
  map_shape[0] = 1;
  map_shape[2] = gt_submap_height_;
  map_shape[3] = gt_submap_width_;

  map_shape[1] = 1;
  label_map->Reshape(map_shape);

  map_shape[1] = 4;
  bbox_map->Reshape(map_shape);

  int offset_x = roi_x;
  int offset_y = roi_y;
  Dtype* label_iter = label_map->mutable_cpu_data();
  Dtype* x_iter = bbox_map->mutable_cpu_data();
  Dtype* y_iter = bbox_map->mutable_cpu_data() + bbox_map->offset(0, 1);
  Dtype* width_iter = bbox_map->mutable_cpu_data() + bbox_map->offset(0, 2);
  Dtype* height_iter = bbox_map->mutable_cpu_data() + bbox_map->offset(0, 3);
  for (int i = 0; i < gt_submap_height_; ++i) {
    for (int j = 0; j < gt_submap_width_; ++j) {
      int gt_idx = FindActiveGT(offset_x, offset_y, gt_bbox);
      if (gt_idx != -1) {
        contain_object = true;

        *label_iter = gt_label[gt_idx];

        if (bbox_anchor_ == BBoxParameter::TOP_LEFT) {
          *x_iter = gt_bbox[gt_idx].x_min() - offset_x;
          *y_iter = gt_bbox[gt_idx].y_min() - offset_y;
          *width_iter = gt_bbox[gt_idx].x_max() - gt_bbox[gt_idx].x_min() + 1;
          *height_iter = gt_bbox[gt_idx].y_max() - gt_bbox[gt_idx].y_min() + 1;
        }
        else if (bbox_anchor_ == BBoxParameter::CENTER) {
          Dtype center_x = (gt_bbox[gt_idx].x_min() + gt_bbox[gt_idx].x_max()) / 2.;
          Dtype center_y = (gt_bbox[gt_idx].y_min() + gt_bbox[gt_idx].y_max()) / 2.;

          *x_iter = center_x - offset_x;
          *y_iter = center_y - offset_y;
          *width_iter = gt_bbox[gt_idx].x_max() - gt_bbox[gt_idx].x_min() + 1;
          *height_iter = gt_bbox[gt_idx].y_max() - gt_bbox[gt_idx].y_min() + 1;
        }
        else
          LOG(FATAL) << "Illegal bbox anchor";

        if (bbox_normalize_) {
          *x_iter /= gt_unit_width_;
          *y_iter /= gt_unit_height_;
          *width_iter /= gt_unit_width_;
          *height_iter /= gt_unit_height_;
        }
      }
      else {
        *label_iter = LabelParameter::NONE;
        *x_iter = BBoxParameter::DUMMY_VALUE;
        *y_iter = BBoxParameter::DUMMY_VALUE;
        *width_iter = BBoxParameter::DUMMY_VALUE;
        *height_iter = BBoxParameter::DUMMY_VALUE;
      }

      offset_x += gt_h_stride_;
      ++label_iter;
      ++x_iter;
      ++y_iter;
      ++width_iter;
      ++height_iter;
    }

    offset_x = roi_x;
    offset_y += gt_v_stride_;
  }

  return contain_object;
}

template <typename Dtype>
int TopSubmapDataLayer<Dtype>::FindActiveGT(
    int offset_x, int offset_y, const std::vector<bgm::BBox<Dtype> >& bbox) const {
  CHECK_GE(offset_x, 0);
  CHECK_GE(offset_y, 0);

  int i;
  for (i = 0; i < bbox.size(); ++i) {
    const bgm::BBox<Dtype>& candidate = bbox[i];

    Dtype center_x = (candidate.x_min() + candidate.x_max()) / 2.0;
    Dtype center_y = (candidate.y_min() + candidate.y_max()) / 2.0;

    if (center_x >= offset_x &&
        center_x <= offset_x + gt_unit_width_ - 1 &&
        center_y >= offset_y &&
        center_y <= offset_y + gt_unit_height_ - 1)
      break;
  }

  return (i == bbox.size() ? -1 : i);
}

template <typename Dtype>
void TopSubmapDataLayer<Dtype>::CopyTop(int batch_idx,
                                        const TopBlob& top_blob,
                                        const std::vector<Blob<Dtype>*>& top) const {
  CHECK_GE(batch_idx, 0);
  CHECK_LT(batch_idx, SUBMAP_BATCH_SIZE_);
  CHECK_GE(top.size(), 1);

  // data
#ifndef NDEBUG
  cv::Mat debug_data, debug_label;
  cv::Mat debug_bbox_x, debug_bbox_y, debug_bbox_w, debug_bbox_h;
  cv::Mat debug_offset_x, debug_offset_y, debug_offset_w, debug_offset_h;
#endif // !NDEBUG

  const Dtype* data_src = top_blob.data.cpu_data();
  Dtype* data_dst = top[0]->mutable_cpu_data() + top[0]->offset(batch_idx);
  caffe_copy(top_blob.data.count(), data_src, data_dst);

#ifndef NDEBUG
  // debug
  int img_width = top_blob.data.width();
  int img_height = top_blob.data.height();
  int img_size = img_width * img_height;
  std::vector<cv::Mat> bgr(3);
  bgr[0] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst);
  bgr[1] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst + img_size);
  bgr[2] = cv::Mat(cv::Size(img_width, img_height), CV_32FC1, data_dst + img_size * 2);
  debug_data = cv::Mat(cv::Size(img_width, img_height), CV_32FC3);
  cv::merge(bgr, debug_data);
  debug_data.convertTo(debug_data, CV_8UC3);
#endif // !NDEBUG

  // offset
  const Dtype* offset_src = top_blob.offset.cpu_data();
  Dtype* offset_dst = top[1]->mutable_cpu_data() + top[1]->offset(batch_idx);
  caffe_copy(top_blob.offset.count(), offset_src, offset_dst);

#ifndef NDEBUG
  // debug
  debug_offset_x = cv::Mat(cv::Size(top_blob.offset.width(),
                                    top_blob.offset.height()),
                            CV_32FC1, offset_dst);
  debug_offset_y = cv::Mat(cv::Size(top_blob.offset.width(),
                                    top_blob.offset.height()),
                            CV_32FC1, offset_dst + offset_submap_width_*offset_submap_height_);
  //debug_offset_w = cv::Mat(cv::Size(top_blob.bbox.width(),
  //                                  top_blob.bbox.height()),
  //                         CV_32FC1, offset_dst + submap_width_*submap_height_ * 2);
  //debug_offset_h = cv::Mat(cv::Size(top_blob.bbox.width(),
  //                                  top_blob.bbox.height()),
  //                         CV_32FC1, offset_dst + submap_width_*submap_height_ * 3);

  //Blob<Dtype>& base1 = const_cast<Blob<Dtype>&>(GetOffsetMap(0, 0));
  //Dtype* base1_x = base1.mutable_cpu_data();
  //Dtype* base1_y = base1.mutable_cpu_data() + base1.offset(0, 1);
  //cv::Mat base1_x_mat(cv::Size(top_blob.bbox.width(),
  //                              top_blob.bbox.height()),
  //                    CV_32FC1, base1_x);
  //cv::Mat base1_y_mat(cv::Size(top_blob.bbox.width(),
  //                              top_blob.bbox.height()),
  //                    CV_32FC1, base1_y);
#endif // !NDEBUG

  // label
  if (top.size() > 2) {
    const Dtype* label_src = top_blob.label.cpu_data();
    Dtype* label_dst = top[2]->mutable_cpu_data() + top[2]->offset(batch_idx);
    caffe_copy(top_blob.label.count(), label_src, label_dst);

#ifndef NDEBUG
    // debug
    debug_label = cv::Mat(cv::Size(top_blob.label.width(),
                                   top_blob.label.height()),
                          CV_32FC1, label_dst);
#endif // !NDEBUG
  }

  // bbox
  if (top.size() > 3) {
    const Dtype* bbox_src = top_blob.bbox.cpu_data();
    Dtype* bbox_dst = top[3]->mutable_cpu_data() + top[3]->offset(batch_idx);
    caffe_copy(top_blob.bbox.count(), bbox_src, bbox_dst);

#ifndef NDEBUG
    // debug
    debug_bbox_x = cv::Mat(cv::Size(top_blob.bbox.width(),
                                    top_blob.bbox.height()),
                           CV_32FC1, bbox_dst);
    debug_bbox_y = cv::Mat(cv::Size(top_blob.bbox.width(),
                                    top_blob.bbox.height()),
                           CV_32FC1, bbox_dst + gt_submap_width_*gt_submap_height_);
    debug_bbox_w = cv::Mat(cv::Size(top_blob.bbox.width(),
                                    top_blob.bbox.height()),
                           CV_32FC1, bbox_dst + gt_submap_width_*gt_submap_height_ * 2);
    debug_bbox_h = cv::Mat(cv::Size(top_blob.bbox.width(),
                                    top_blob.bbox.height()),
                           CV_32FC1, bbox_dst + gt_submap_width_*gt_submap_height_ * 3);
#endif // !NDEBUG
  }
}

#ifdef CPU_ONLY
STUB_GPU(TopSubmapDataLayer);
#endif

INSTANTIATE_CLASS(TopSubmapDataLayer);
REGISTER_LAYER_CLASS(TopSubmapData);
} // namespace caffe