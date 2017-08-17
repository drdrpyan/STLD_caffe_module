#include "bbox_anno_map_layer.hpp"

#include <limits>

namespace caffe
{

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BBoxAnnoMapParameter param = 
      this->layer_param_.bbox_anno_map_param();
  reception_field_height_ = param.receptive_field_hight();
  reception_field_width_ = param.receptive_field_width();
  horizontal_stride_ = param.horizontal_stride();
  vertical_stride_ = param.vertical_stride();

  num_label_ = this->layer_param_.label_param().num_label();

  CHECK(reception_field_height_ > 0 && reception_field_width_ > 0) <<
    "Invalid reception_field_size. (h, w) = (" <<
    reception_field_height_ << ", " << reception_field_width_ << ")";
  CHECK_GT(horizontal_stride_, 0) << 
    "Horizontal stride is lessor or equal than 0";
  CHECK_GT(vertical_stride_, 0) << 
    "Vertical stride is lessor or equal than 0";
  CHECK_GT(num_label_, 0) <<
    "Number of labels is lesser or equal than 0";
}

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& input = *(bottom[0]);
  Blob<Dtype>& output_labelmap = *(top[0]);
  Blob<Dtype>& output_bboxmap = *(top[1]);

  int output_h, output_w;
  ComputeOutputHW(input.height(), input.width(),
                  &output_h, &output_w);

  std::vector<int> labelmap_shape(4);
  labelmap_shape[0] = input.num();
  labelmap_shape[1] = num_label_;
  labelmap_shape[2] = output_h;
  labelmap_shape[3] = output_w;

  std::vector<int> bboxmap_shape(4);
  bboxmap_shape[0] = input.num();
  bboxmap_shape[1] = 4; // min_x, min_y, max_x, max_y
  bboxmap_shape[2] = output_h;
  bboxmap_shape[3] = output_w;

  output_labelmap.Reshape(labelmap_shape);
  output_bboxmap.Reshape(bboxmap_shape);
}

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::InitMapShape(
    const BBoxAnnoMapParameter& param, int num_label) {
  CHECK(IsValidParam(param));
  CHECK(num_label > 0);

  int height = 
    ((param.img_height() - param.reception_field_hight()) / 
      param.h_stride())
    + 1;
  int width = 
    ((param.img_width() - param.reception_field_w()) / 
      param.w_stride())
    + 1;

  label_map_shape_.resize(4);
  label_map_shape_
}

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& input = *(bottom[0]);
  Blob<Dtype>& label_map = *(top[0]);
  Blob<Dtype>& bbox_map = *(top[1]);

  CHECK(IsValidInputBlob(input));

  vector<vector<BBoxAnno> > parsed_input;
  ParseInputBlob(input, &parsed_input);

  for (int n = 0; n < input.num(); n++) {
    Dtype* label_map_ptr = label_map.mutable_cpu_data();
    label_map_ptr += label_map.offset(n);

    Dtype* bbox_map_ptr = bbox_map.mutable_cpu_data();
    bbox_map_ptr += bbox_map.offset(n);

    MakeMaps(parsed_input[n],
             label_map.height(), label_map.width(),
             label_map_ptr, bbox_map_ptr);
  }
}

// 임시 구현
template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::ComputeOutputHW(
    int input_h, int input_w,
    int* output_h, int* output_w) const {
  CHECK(output_h && output_w);

  CHECK_GE(input_h, reception_field_height_) <<
    "Input is smaller than reception field : " <<
    "input width = " << input_h <<
    ", reception field width = " << input_w;
  CHECK_GE(input_w, reception_field_width_) <<
    "Input is smaller than reception field : " <<
    "input height = " << input_h <<
    ", reception field height = " << input_w;

  int rest_h = input_h - reception_field_height_;
  int rest_w = input_w - reception_field_width_;
  *output_h = rest_h / vertical_stride_ + 1;
  *output_w = rest_w / horizontal_stride_ + 1;
}

template <typename Dtype>
int BBoxAnnoMapLayer<Dtype>::FindBestBBoxAnno(
    const BBox& receptive_field,
    const vector<BBoxAnno>& candidates) const {
  int index = -1;
  float distance = std::numeric_limits<float>::max();

  for (int i = 0; i < candidates.size(); i++) {
    if (IsBBoxInReceptiveField(receptive_field,
                               candidates[i].second)) {
      float new_distance = 
        ComputeCenterDistance(receptive_field,
                              candidates[i].second);
      if (new_distance < distance) {
        index = i;
        distance = new_distnace;
      }
    }
  }

  return index;
}

template <typename Dtype>
bool BBoxAnnoMapLayer<Dtype>::IsValidInputBlob(
    const Blob<Dtype>& input_blob) const {
  if (input_blob.num() < 0)
    return false;
  if (input_blob.channels() != 1)
    return false;
  if (input_blob.height() < 0)
    return false;
  if (input_blob.width != 5)
    return false;

  return true;
}

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::ParseInputBlob(
    const Blob<Dtype>& input_blob,
    vector<vector<BBoxAnno > >* bbox_anno) const {
  CHECK(bbox_anno);
  
  bbox_anno->resize(input_blob.num());
  
  Dtype const * input_blob_data = input_blob.cpu_data();
  //vector<vector<BBoxAnno<Dtype> > >::iterator bbox_anno_vec_iter = 
  //    bbox_anno->back();
  auto bbox_anno_vec_iter = bbox_anno->back();
  for (int n = input_blob.num(); n--; ) {
    bbox_anno_vec_iter->clear();
    for (int h = input_blob.height(); h--; ) {
      Dtype label = *input_blob_data++;
      if (label != LabelParameter::DUMMY_LABEL) {
        Dtype x_min = *input_blob_data++;
        Dtype y_min = *input_blob_data++;
        Dtype x_max = *input_blob_data++;
        Dtype y_max = *input_blob_data++;

        bbox_anno_vec_iter->push_back(
            BBoxAnno(label, 
                     bgm::BBox<Dtype>(x_min, y_min, 
                                      x_max, y_max));
      }
      else {
        input_blob_data += 4;
      }
    }
    bbox_anno_vec_iter++;
  }
}

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::MakeMaps(
    const vector<BBoxAnno>& bbox_anno,
    int map_height, int map_width,
    Dtype* label_map,  Dtype* bbox_map) const {
  CHECK(map_height > 0 && map_width);
  CHECK(label_map && bbox_map);

  bgm::BBox<Dtype> receptive_field(0, 0,
                                   receptive_field_width_ - 1,
                                   receptive_field_height_ - 1);

  int map_size = map_height * map_width;
  Dtype* label_iter = label_map;
  Dtype* bbox_x_min_iter = bbox_map;
  Dtype* bbox_y_min_iter = bbox_x_min_iter + map_size;
  Dtype* bbox_x_max_iter = bbox_y_min_iter + map_size;
  Dtype* bbox_y_max_iter = bbox_x_max_iter + map_size;

  for (int i = map_height; i--; ) {    
    for (int j = map_width; j--; ) {
      int best_idx = FindBestBBoxAnno(receptive_field,
                                      bbox_anno);
      if (best_idx != -1) {
        const BBoxAnno& best = bbox_anno[best_idx];
        *label_iter++ = best.first;
        *bbox_x_min_iter++ = best.second.x_min();
        *bbox_y_min_iter++ = best.second.y_min();
        *bbox_x_max_iter++ = best.second.x_max();
        *bbox_y_max_iter++ = best.second.y_max();
      }
      else {
        *label_iter++ = -1;
        *bbox_x_min_iter++ = -1;
        *bbox_y_min_iter++ = -1;
        *bbox_x_max_iter++ = -1;
        *bbox_y_max_iter++ = -1;
      }

      receptive_field.ShiftX(horizontal_stride_);
    }

    receptive_field.set_x_min(0);
    receptive_field.set_x_max(receptive_field_width_ - 1);
    receptive_field.ShiftY(vertical_stride_);
  }
}

template <typename Dtype>
bool BBoxAnnoMapLayer<Dtype>::IsBBoxInReceptiveField(
    const bgm::BBox<Dtype>& receptive_field,
    const bgm::BBox<Dtype>& obj_bbox) const {
  bool cond1 = receptive_field.x_min() <= obj_bbox.x_min();
  bool cond2 = receptive_field.y_min() <= obj_bbox.y_min();
  bool cond3 = receptive_field.x_max() >= obj_bbox.x_max();
  bool cond4 = receptive_field.y_max() >= obj_bbox.y_max();
  return cond1 && cond2 && cond3 && cond4;
}

template <typename Dtype>
float BBoxAnnoMapLayer<Dtype>::ComputeCenterDistance(
    const bgm::BBox<Dtype>& bbox1, 
    const bgm::BBox<Dtype>& bbox2) const {
  float bbox1_x_mid = (bbox1.x_min() + bbox1.x_max()) / 2.0f;
  float bbox1_y_mid = (bbox1.y_min() + bbox1.y_max()) / 2.0f;
  float bbox2_x_mid = (bbox2.x_min() + bbox2.x_max()) / 2.0f;
  float bbox2_y_mid = (bbox2.y_min() + bbox2.y_max()) / 2.0f;
  float term1 = std::powf(bbox1_x_mid - bbox2_x_mid, 2.0f);
  float term2 = std::powf(bbox1_y_mid - bbox2_y_mid, 2.0f);
  float dist = std::sqrtf(term1 + term2);
  return dist;
}

} // namespace caffe