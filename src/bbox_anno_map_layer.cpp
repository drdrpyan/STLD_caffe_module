#include "bbox_anno_map_layer.hpp"

#include <limits>

namespace caffe
{

template <typename Dtype>
inline BBoxAnnoMapLayer<Dtype>::BBoxAnnoMapLayer(
    const LayerParameter& param) 
  : Layer<Dtype>(param),
    NUM_LABEL_(param.label_param().num_label()),
    IMG_HEIGHT_(param.bbox_anno_map_param().img_height()),
    IMG_WIDTH_(param.bbox_anno_map_param().img_width()),
    RECEPTIVE_FIELD_HEIGHT_(param.bbox_anno_map_param().receptive_field_height()),
    RECEPTIVE_FIELD_WIDTH_(param.bbox_anno_map_param().receptive_field_width()),
    VERTICAL_STRIDE_(param.bbox_anno_map_param().vertical_stride()),
    HORIZONTAL_STRIDE_(param.bbox_anno_map_param().horizontal_stride()),
    NORMALIZED_POSITION_IN_(false), NORMALIZED_POSITION_OUT_(true) {
  CHECK_GT(NUM_LABEL_, 0) << "Invalid the number of labels";
  CHECK_GT(IMG_HEIGHT_, 0) << "Invalid parameter : image height";
  CHECK_GT(IMG_WIDTH_, 0) << "Ivalid parameter : image width";
  CHECK(RECEPTIVE_FIELD_HEIGHT_ > 0 && RECEPTIVE_FIELD_HEIGHT_ <= IMG_HEIGHT_) <<
    "Invalid parameter : receptive field height";
  CHECK(RECEPTIVE_FIELD_WIDTH_ > 0 && RECEPTIVE_FIELD_WIDTH_ <= IMG_WIDTH_) <<
    "Invalid parameter : receptive field width";
  CHECK_GT(VERTICAL_STRIDE_, 0) << "Invalid parameter : vertical stride";
  CHECK_GT(HORIZONTAL_STRIDE_, 0) << "Invalid parameter : horizontal stride";
}

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& input = *(bottom[0]);
  Blob<Dtype>& output_labelmap = *(top[0]);
  Blob<Dtype>& output_bboxmap = *(top[1]);

  labelmap_shape_[0] = input.num();
  output_labelmap.Reshape(labelmap_shape_);

  bboxmap_shape_[0] = input.num();
  output_bboxmap.Reshape(bboxmap_shape_);
}

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& input = *(bottom[0]);
  Blob<Dtype>& label_map = *(top[0]);
  Blob<Dtype>& bbox_map = *(top[1]);

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
void BBoxAnnoMapLayer<Dtype>::InitMapShape(
    const LabelParameter& label_param,
    const BBoxAnnoMapParameter& bbox_anno_map_param) {
  int map_height, map_width;
  ComputeMapHW(&map_height, &map_width);

  labelmap_shape_.resize(4);
  labelmap_shape_[1] = NUM_LABEL_;
  labelmap_shape_[2] = map_height;
  labelmap_shape_[3] = map_width;

  bboxmap_shape_.resize(4);
  bboxmap_shape_[1] = 4; // min_x, min_y, max_x, max_y
  bboxmap_shape_[2] = map_height;
  bboxmap_shape_[3] = map_width;
}

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::ComputeMapHW(
    int *map_height, int *map_width) const {
  CHECK(map_height && map_width);
  int rest_h = IMG_HEIGHT_ - RECEPTIVE_FIELD_HEIGHT_;
  int rest_w = IMG_WIDTH_ - RECEPTIVE_FIELD_WIDTH_;
  *map_height = rest_h / VERTICAL_STRIDE_ + 1;
  *map_width = rest_w / HORIZONTAL_STRIDE_ + 1;
}

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::MakeMaps(
    const vector<BBoxAnno>& bbox_anno,
    int map_height, int map_width,
    Dtype* label_map,  Dtype* bbox_map) const {
  CHECK(map_height > 0 && map_width);
  CHECK(label_map && bbox_map);

  bgm::BBox<Dtype> receptive_field(0, 0,
                                   RECEPTIVE_FIELD_WIDTH_ - 1,
                                   RECEPTIVE_FIELD_HEIGHT_ - 1);

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
        bgm::BBox<Dtype> relocated_bbox;
        RelocateBBox(receptive_field, best.second, &relocated_bbox);
        *label_iter++ = best.first;
        *bbox_x_min_iter++ = relocated_bbox.x_min();
        *bbox_y_min_iter++ = relocated_bbox.y_min();
        *bbox_x_max_iter++ = relocated_bbox.x_max();
        *bbox_y_max_iter++ = relocated_bbox.y_max();
      }
      else {
        *label_iter++ = LabelParameter::NONE;
        *bbox_x_min_iter++ = -1;
        *bbox_y_min_iter++ = -1;
        *bbox_x_max_iter++ = -1;
        *bbox_y_max_iter++ = -1;
      }

      receptive_field.ShiftX(HORIZONTAL_STRIDE_);
    }

    receptive_field.set_x_min(0);
    receptive_field.set_x_max(RECEPTIVE_FIELD_WIDTH_ - 1);
    receptive_field.ShiftY(VERTICAL_STRIDE_);
  }
}

template <typename Dtype>
int BBoxAnnoMapLayer<Dtype>::FindBestBBoxAnno(
    const bgm::BBox<Dtype>& receptive_field,
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
        distance = new_distance;
      }
    }
  }

  return index;
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
  auto bbox_anno_vec_iter = bbox_anno->begin();
  for (int n = input_blob.num(); n--; ) {
    bbox_anno_vec_iter->clear();
    for (int h = input_blob.height(); h--; ) {
      Dtype label = *input_blob_data++;
      CHECK(label > 0 && label <= NUM_LABEL_) << "Invalide label";

      if (label != LabelParameter::DUMMY_LABEL) {
        Dtype x_min = *input_blob_data++;
        Dtype y_min = *input_blob_data++;
        Dtype x_max = *input_blob_data++;
        Dtype y_max = *input_blob_data++;

        CHECK(x_min >= 0 && x_min < IMG_WIDTH_) << "Invalid bounding box";
        CHECK(y_min >= 0 && y_min < IMG_HEIGHT_) << "Invalid bounding box";
        CHECK(x_max >= 0 && x_max < IMG_WIDTH_) << "Invalid bounding box";
        CHECK(y_max >= 0 && y_max < IMG_HEIGHT_) << "Invalid bounding box";

        bbox_anno_vec_iter->push_back(
            BBoxAnno(label, 
                     bgm::BBox<Dtype>(x_min, y_min, 
                                      x_max, y_max)));
      }
      else {
        input_blob_data += 4;
      }
    }
    bbox_anno_vec_iter++;
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

template <typename Dtype>
void BBoxAnnoMapLayer<Dtype>::RelocateBBox(
    const bgm::BBox<Dtype>& receptive_field,
    const bgm::BBox<Dtype>& global_position,
    bgm::BBox<Dtype>* local_position) const {
  CHECK(local_position);
  
  *local_position = global_position;
  if (!NORMALIZED_POSITION_IN_)
    local_position->Scale(IMG_WIDTH_, IMG_HEIGHT_,
                          bgm::BBox<Dtype>::ScalePivot::SCENE_TOPLEFT);

  local_position->Shift(receptive_field.x_min(),
                        receptive_field.y_min());

  if (NORMALIZED_POSITION_OUT_)
    local_position->Scale(static_cast<Dtype>(1) / IMG_WIDTH_,
                          static_cast<Dtype>(1) / IMG_HEIGHT_,
                          bgm::BBox<Dtype>::ScalePivot::SCENE_TOPLEFT);
}

INSTANTIATE_CLASS(BBoxAnnoMapLayer);
REGISTER_LAYER_CLASS(BBoxAnnoMap);

} // namespace caffe