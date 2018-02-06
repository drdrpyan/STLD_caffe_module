#include "yololike_loss_layer.hpp"

namespace caffe
{
template <typename Dtype>
YOLOLikeLossLayer<Dtype>::YOLOLikeLossLayer(const LayerParameter& param) 
  : LossLayer<Dtype>(param),
    NUM_BBOX_PER_CELL_(param.yololike_loss_param().num_bbox_per_cell()),
    NUM_CLASS_(param.yololike_loss_param().num_class()),
    NOOBJ_SCALE_(param.yololike_loss_param().noobj_scale()),
    OBJ_SCALE_(param.yololike_loss_param().obj_scale()),
    CLASS_SCALE_(param.yololike_loss_param().class_scale()),
    COORD_SCALE_(param.yololike_loss_param().coord_scale()) {

}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
  const YOLOLikeLossParameter& param = layer_param_.yololike_loss_param();

  if (param.has_bbox_param() && param.bbox_param().has_anchor())
    bbox_anchor_ = param.bbox_param().anchor();
  else
    bbox_anchor_ = BBoxParameter::TOP_LEFT;

  class_weight_.resize(NUM_CLASS_, 1);
  const float* class_weight = param.class_weight().data();
  int num_class_weight = param.class_weight().size();
  if(NUM_CLASS_ < num_class_weight)
    LOG(WARNING) << "# of class weights should be less or equal than # of class";
  std::copy(class_weight, class_weight + std::min(NUM_CLASS_, num_class_weight),
            &(class_weight_[0]));
}
template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  if (bottom.size() == 3) {
    CHECK_EQ(bottom[0]->num(), bottom[2]->num());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
  }
  CHECK_EQ(bottom[0]->channels(), 5 * NUM_BBOX_PER_CELL_ + NUM_CLASS_);

  std::vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);

  grid_width_ = bottom[0]->width();
  grid_height_ = bottom[1]->height();
  num_cells_ = grid_width_ * grid_height_;
  diff_.ReshapeLike(*bottom[0]);

  if (top.size() > 1) // noobj loss
    top[1]->Reshape(loss_shape);
  if (top.size() > 2) // obj loss
    top[2]->Reshape(loss_shape);
  if (top.size() > 3) // class loss
    top[3]->Reshape(loss_shape);
  if (top.size() > 4) // coord loss
    top[4]->Reshape(loss_shape);
  if (top.size() > 5) // area loss
    top[5]->Reshape(loss_shape);
  if (top.size() > 6) // noobj conf
    top[6]->Reshape(loss_shape);
  if (top.size() > 7) // obj conf
    top[7]->Reshape(loss_shape);
  if (top.size() > 8) // class conf
    top[8]->Reshape(loss_shape);
  if (top.size() > 9) // pos class conf
    top[9]->Reshape(loss_shape);
  if (top.size() > 10) // iou
    top[10]->Reshape(loss_shape);
  if (top.size() > 11) // neg conf stddev
    top[11]->Reshape(loss_shape);
  if (top.size() > 12) // neg conf stddev
    top[12]->Reshape(loss_shape);
}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& input = *(bottom[0]);
  const Blob<Dtype>* true_label;
  const Blob<Dtype>* true_bbox;
  GetGT(bottom, &true_label, &true_bbox);


  Clear();
  //noobj_conf_list_.resize(input.num()*input.width()*input.height()*NUM_BBOX_PER_CELL_);
  
  std::vector<Dtype> bbox(4);

  for (int n = 0; n < input.num(); ++n) {
    const Dtype* true_label_iter = true_label->cpu_data() + true_label->offset(n);
    const Dtype* true_x_iter = true_bbox->cpu_data() + true_bbox->offset(n, 0);
    const Dtype* true_y_iter = true_bbox->cpu_data() + true_bbox->offset(n, 1);
    const Dtype* true_w_iter = true_bbox->cpu_data() + true_bbox->offset(n, 2);
    const Dtype* true_h_iter = true_bbox->cpu_data() + true_bbox->offset(n, 3);

    for (int h = 0; h < input.height(); ++h) {
      for (int w = 0; w < input.width(); ++w) {
        if (*true_label_iter == LabelParameter::NONE) {
          noobj_count_++;
          ForwardNegative(input, n, h, w);
        }
        else if(*true_label_iter != -1) {
          obj_count_++;

          bbox[BBoxAttr::X] = *true_x_iter;
          bbox[BBoxAttr::Y] = *true_y_iter;
          bbox[BBoxAttr::W] = *true_w_iter;
          bbox[BBoxAttr::H] = *true_h_iter;

          ForwardPositive(input, n, h, w, *true_label_iter, bbox);
        }

        true_label_iter++;
        true_x_iter++;
        true_y_iter++;
        true_w_iter++;
        true_h_iter++;
      }
    }
  }

  if (noobj_count_ > 0)
    noobj_loss_ /= noobj_count_;
  if (obj_count_ > 0) {
    obj_loss_ /= obj_count_;
    class_loss_ /= obj_count_;
    coord_loss_ /= obj_count_;
    area_loss_ /= obj_count_;
  }
  Dtype loss = noobj_loss_ + obj_loss_ +  class_loss_ + coord_loss_ + area_loss_;
  top[0]->mutable_cpu_data()[0] = loss;

  if (noobj_count_ > 0)
    noobj_conf_ /= noobj_count_;
  if (obj_count_ > 0) {
    obj_conf_ /= obj_count_;
    neg_class_conf_ /= (obj_count_ * (NUM_CLASS_-1));
    pos_class_conf_ /= obj_count_;
    iou_ /= obj_count_;
  }

  // compute std_dev
  Dtype neg_std_dev = 0;
  if (noobj_count_ > 1) {
    for (int i = 0; i < noobj_conf_list_.size(); ++i)
      neg_std_dev += std::pow((noobj_conf_list_[i] - noobj_conf_), 2);
    neg_std_dev /= (noobj_conf_list_.size() - 1);
    neg_std_dev = std::sqrt(neg_std_dev);
  }
  Dtype pos_std_dev = 0;
  if (obj_count_ > 1) {
    for (int i = 0; i < obj_conf_list_.size(); ++i)
      pos_std_dev += std::pow((obj_conf_list_[i] - obj_conf_), 2);
    pos_std_dev /= (obj_conf_list_.size() - 1);
    pos_std_dev = std::sqrt(pos_std_dev);
  }
  
  //LOG(INFO) << "loss: " << loss << " noobj_loss: " << noobj_loss_
  //  << " obj_loss: " << obj_loss_ << " class_loss: " << class_loss_
  //  << " coord_loss: " << coord_loss << " area_loss: " << area_loss_;

  //LOG(INFO) << "noobj_conf: " << noobj_conf_ << " obj_conf: " << obj_conf_
  //  << " class_conf: " << class_conf_ << "pos_class_conf: " << pos_class_conf_
  //  << " iou: " << iou_;

  if (top.size() > 1) // noobj loss
    top[1]->mutable_cpu_data()[0] = noobj_loss_;
  if (top.size() > 2) // obj loss
    top[2]->mutable_cpu_data()[0] = obj_loss_;
  if (top.size() > 3) // class loss
    top[3]->mutable_cpu_data()[0] = class_loss_;
  if (top.size() > 4) // coord loss
    top[4]->mutable_cpu_data()[0] = coord_loss_;
  if (top.size() > 5) // area loss
    top[5]->mutable_cpu_data()[0] = area_loss_;
  if (top.size() > 6) // noobj conf
    top[6]->mutable_cpu_data()[0] = noobj_conf_;
  if (top.size() > 7) // obj conf
    top[7]->mutable_cpu_data()[0] = obj_conf_;
  if (top.size() > 8) // class conf
    top[8]->mutable_cpu_data()[0] = neg_class_conf_;
  if (top.size() > 9) // pos class conf
    top[9]->mutable_cpu_data()[0] = pos_class_conf_;
  if (top.size() > 10) // iou
    top[10]->mutable_cpu_data()[0] = iou_;
  if (top.size() > 11) // iou
    top[11]->mutable_cpu_data()[0] = neg_std_dev;
  if (top.size() > 12) // iou
    top[12]->mutable_cpu_data()[0] = pos_std_dev;
}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_cpu_axpby(bottom[0]->count(), alpha, diff_.cpu_data(),
                    Dtype(0), bottom[0]->mutable_cpu_diff());
  }
}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::GetGT(const vector<Blob<Dtype>*>& bottom,
                                     const Blob<Dtype>** true_label,
                                     const Blob<Dtype>** true_bbox) const {
  CHECK(true_label);
  CHECK(true_bbox);

  if (bottom.size() == 3) {
    *true_label = bottom[1];
    *true_bbox = bottom[2];
  }
  else
    LOG(FATAL) << "Not implemented yet.";
}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::Clear() {
  obj_count_ = 0;
  noobj_count_ = 0;

  noobj_loss_ = 0;
  obj_loss_ = 0;
  class_loss_ = 0;
  coord_loss_ = 0;
  area_loss_ = 0;
  noobj_conf_ = 0;
  obj_conf_ = 0;
  neg_class_conf_ = 0;
  pos_class_conf_ = 0;
  iou_ = 0;

  noobj_conf_list_.clear();
  obj_conf_list_.clear();
}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::ForwardNegative(
    const Blob<Dtype>& input, int n, int h, int w) {
  const Dtype* input_data = input.cpu_data();
  Dtype* diff_data = diff_.mutable_cpu_data();

  for (int b = 0; b < NUM_BBOX_PER_CELL_; ++b) {
    int bbox_conf_offset = BBoxOffset(input, n, b, BBoxAttr::CONF, h, w);
    ComputeNoobjLossDiff(bbox_conf_offset, input_data, diff_data);
  }
}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::ForwardPositive(
    const Blob<Dtype>& input, int n, int h, int w,
    Dtype true_label, const std::vector<Dtype>& true_bbox) {
  const Dtype* input_data = input.cpu_data();
  Dtype* diff_data = diff_.mutable_cpu_data();

  for (int c = 1; c <= NUM_CLASS_; ++c) {
    int class_conf_offset = ClassConfOffset(input, n, c, h, w);
    ComputeClassConfLossDiff(class_conf_offset, c, true_label,
                             input_data, diff_data);
  }

  int best_bbox_idx;
  Dtype best_iou;
  std::vector<int> best_bbox_offsets;
  FindBestBBox(input, n, h, w, true_bbox, 
               &best_bbox_idx, &best_iou, &best_bbox_offsets);
  for (int b = 0; b < NUM_BBOX_PER_CELL_; ++b) {
    int bbox_conf_offset = BBoxOffset(input, n, b, BBoxAttr::CONF, h, w);

    if (b != best_bbox_idx)
      ComputeNoobjLossDiff(bbox_conf_offset,
                           input_data, diff_data);
    else {
      ComputeObjLossDiff(bbox_conf_offset, best_iou,
                         input_data, diff_data);
    }
  }

  ComputeCoordAreaLossDiff(true_bbox, best_bbox_offsets,
                           input_data, diff_data);
}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::ComputeNoobjLossDiff(
    int bbox_conf_offset, const Dtype* input_data, Dtype* diff_data) {
  CHECK(input_data);
  CHECK(diff_data);

  Dtype input_bbox_conf = *(input_data + bbox_conf_offset);
  //noobj_loss_ += NOOBJ_SCALE_ * std::pow(input_bbox_conf - 0, 2);
  //*(diff_data + bbox_conf_offset) = NOOBJ_SCALE_ * (input_bbox_conf - 0);
  //if (input_bbox_conf > 0.) {
  //  //noobj_loss_ += NOOBJ_SCALE_ * std::pow(input_bbox_conf + 0.5, 2);
  //  noobj_loss_ += NOOBJ_SCALE_ * std::abs(input_bbox_conf - 0);
  //  *(diff_data + bbox_conf_offset) = NOOBJ_SCALE_ * (input_bbox_conf - 0);
  //}

  noobj_loss_ += NOOBJ_SCALE_ * std::abs(input_bbox_conf - 0);
  *(diff_data + bbox_conf_offset) = NOOBJ_SCALE_ * (input_bbox_conf - 0);

  noobj_conf_ += input_bbox_conf;

  noobj_conf_list_.push_back(input_bbox_conf);
}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::ComputeObjLossDiff(
    int bbox_conf_offset, Dtype best_iou,
    const Dtype* input_data, Dtype* diff_data) {
  CHECK(input_data);
  CHECK(diff_data);

  Dtype input_bbox_conf = *(input_data + bbox_conf_offset);
  //if (std::isnan(input_bbox_conf)) input_bbox_conf = 0;
  //if (input_bbox_conf > 2.) {
  //  //obj_loss_ += OBJ_SCALE_ * std::pow(input_bbox_conf - 1.5, 2);
  //  obj_loss_ += OBJ_SCALE_ * std::abs(input_bbox_conf - 1);
  //  *(diff_data + bbox_conf_offset) = OBJ_SCALE_ * (input_bbox_conf - 1.);
  //}
  //else if (input_bbox_conf < 1.) {
  //  //obj_loss_ += OBJ_SCALE_ * std::pow(input_bbox_conf - 1.5, 2);
  //  obj_loss_ += OBJ_SCALE_ * std::abs(input_bbox_conf - 1);
  //  //*(diff_data + bbox_conf_offset) = OBJ_SCALE_ * (input_bbox_conf - best_iou);
  //  *(diff_data + bbox_conf_offset) = OBJ_SCALE_ * (input_bbox_conf - 1.);
  //}

  //obj_loss_ += OBJ_SCALE_ * std::abs(input_bbox_conf - 1);
  //*(diff_data + bbox_conf_offset) = OBJ_SCALE_ * (input_bbox_conf - 1);
  obj_loss_ += OBJ_SCALE_ * std::abs(input_bbox_conf - best_iou);
  *(diff_data + bbox_conf_offset) = OBJ_SCALE_ * (input_bbox_conf - 1);

  iou_ += best_iou;
  obj_conf_ += input_bbox_conf;

  obj_conf_list_.push_back(input_bbox_conf);
}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::ComputeClassConfLossDiff(
    int class_conf_offset, Dtype prediction, Dtype true_label,
    const Dtype* input_data, Dtype* diff_data) {
  CHECK(input_data);
  CHECK(diff_data);

  Dtype class_conf = *(input_data + class_conf_offset);
  //if (std::isnan(class_conf)) class_conf = 0;

  bool positive = (prediction == true_label);
  Dtype target = positive ? 1 : 0;

  Dtype scale = CLASS_SCALE_ * ClassWeight(true_label);
  if (positive) {
    if ((class_conf < target) || class_conf > (target+1)) {
      //class_loss_ += scale * std::pow(class_conf - 1, 2);
      class_loss_ += scale * std::abs(class_conf - 1);
      *(diff_data + class_conf_offset) = scale * (class_conf - target);
    }
  }
  else {
    if ((class_conf > target) || (class_conf < (target - 1))) {
      //class_loss_ += scale * std::pow(class_conf - 0, 2);
      class_loss_ += scale * std::abs(class_conf - 0);
      *(diff_data + class_conf_offset) = scale * (class_conf - 0);
    }
  }

  if(positive)
    pos_class_conf_ += class_conf;
  else
    neg_class_conf_ += class_conf;    
}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::ComputeCoordAreaLossDiff(
    const std::vector<Dtype>& true_bbox,
    const std::vector<int>& best_bbox_offsets,
    const Dtype* input_data, Dtype* diff_data) {
  CHECK_EQ(true_bbox.size(), 4);
  CHECK_EQ(best_bbox_offsets.size(), 4);
  CHECK(input_data);
  CHECK(diff_data);

  Dtype best_x = *(input_data + best_bbox_offsets[BBoxAttr::X]);
  //if (std::isnan(best_x)) best_x = 0;
  Dtype best_y = *(input_data + best_bbox_offsets[BBoxAttr::Y]);
  //if (std::isnan(best_y)) best_y = 0;
  //Dtype best_w 
  //    = std::sqrt(*(input_data + best_bbox_offsets[BBoxAttr::W]));
  Dtype best_w = *(input_data + best_bbox_offsets[BBoxAttr::W]);
  Dtype best_w_sqrt = (best_w < 0) ? std::sqrt(-best_w) : std::sqrt(best_w);
  //if (std::isnan(best_w)) best_w = 0;
  //Dtype best_h 
  //    = std::sqrt(*(input_data + best_bbox_offsets[BBoxAttr::H]));
  Dtype best_h = *(input_data + best_bbox_offsets[BBoxAttr::H]);
  Dtype best_h_sqrt = (best_h < 0) ? std::sqrt(-best_h) : std::sqrt(best_h);
  //if (std::isnan(best_h)) best_h = 0;

  Dtype true_w_sqrt = std::sqrt(true_bbox[BBoxAttr::W]);
  Dtype true_h_sqrt = std::sqrt(true_bbox[BBoxAttr::H]);

  //coord_loss_ += COORD_SCALE_ * std::pow(best_x - true_bbox[BBoxAttr::X], 2);
  //coord_loss_ += COORD_SCALE_ * std::pow(best_y - true_bbox[BBoxAttr::Y], 2);
  coord_loss_ += COORD_SCALE_ * std::abs(best_x - true_bbox[BBoxAttr::X]);
  coord_loss_ += COORD_SCALE_ * std::abs(best_y - true_bbox[BBoxAttr::Y]);
  area_loss_ += COORD_SCALE_ * std::pow(best_w_sqrt - true_w_sqrt, 2);
  area_loss_ += COORD_SCALE_ * std::pow(best_h_sqrt - true_h_sqrt, 2);
  //area_loss_ += COORD_SCALE_ * std::pow(best_w - true_bbox[BBoxAttr::W], 2);
  //area_loss_ += COORD_SCALE_ * std::pow(best_h - true_bbox[BBoxAttr::H], 2);

  *(diff_data + best_bbox_offsets[BBoxAttr::X]) 
      = COORD_SCALE_ * (best_x - true_bbox[BBoxAttr::X]);
  *(diff_data + best_bbox_offsets[BBoxAttr::Y]) 
      = COORD_SCALE_ * (best_y - true_bbox[BBoxAttr::Y]);
  *(diff_data + best_bbox_offsets[BBoxAttr::W]) 
      = COORD_SCALE_ * (best_w - true_bbox[BBoxAttr::W]);
  *(diff_data + best_bbox_offsets[BBoxAttr::H]) 
      = COORD_SCALE_ * (best_h - true_bbox[BBoxAttr::H]);
  //*(diff_data + best_bbox_offsets[BBoxAttr::W]) 
  //    = COORD_SCALE_ * (best_w - true_bbox[BBoxAttr::W]);
  //*(diff_data + best_bbox_offsets[BBoxAttr::H]) 
  //    = COORD_SCALE_ * (best_h - true_bbox[BBoxAttr::H]);

}

template <typename Dtype>
void YOLOLikeLossLayer<Dtype>::FindBestBBox(
    const Blob<Dtype>& input, int n, int h, int w,
    const std::vector<Dtype>& true_bbox,
    int* best_bbox_idx, Dtype* best_iou,
    std::vector<int>* best_bbox_offsets) {
  CHECK_EQ(true_bbox.size(), 4);
  CHECK(best_bbox_idx);
  CHECK(best_bbox_offsets);

  best_bbox_offsets->resize(4);

  const Dtype* input_data = input.cpu_data();

  std::vector<Dtype> bbox(4);
  *best_iou = 0.;
  Dtype best_rmse = std::numeric_limits<float>::max();
  for (int i = 0; i < NUM_BBOX_PER_CELL_; ++i) {
    int x_offset = BBoxOffset(input, n, i, BBoxAttr::X, h, w);
    int y_offset = BBoxOffset(input, n, i, BBoxAttr::Y, h, w);
    int w_offset = BBoxOffset(input, n, i, BBoxAttr::W, h, w);
    int h_offset = BBoxOffset(input, n, i, BBoxAttr::H, h, w);

    bbox[BBoxAttr::X] = *(input_data + x_offset);
    bbox[BBoxAttr::Y] = *(input_data + y_offset);
    bbox[BBoxAttr::W] = *(input_data + w_offset);
    bbox[BBoxAttr::H] = *(input_data + h_offset);

    Dtype iou = CalcIoU(bbox, true_bbox);
    Dtype rmse = CalcRMSE(bbox, true_bbox);

    if (*best_iou > 0 || iou > 0) {
      if (iou > *best_iou) {
        *best_iou = iou;

        *best_bbox_idx = i;
        (*best_bbox_offsets)[BBoxAttr::X] = x_offset;
        (*best_bbox_offsets)[BBoxAttr::Y] = y_offset;
        (*best_bbox_offsets)[BBoxAttr::W] = w_offset;
        (*best_bbox_offsets)[BBoxAttr::H] = h_offset;
      }
    }
    else {
      if (rmse < best_rmse) {
        best_rmse = rmse;

        *best_bbox_idx = i;
        (*best_bbox_offsets)[BBoxAttr::X] = x_offset;
        (*best_bbox_offsets)[BBoxAttr::Y] = y_offset;
        (*best_bbox_offsets)[BBoxAttr::W] = w_offset;
        (*best_bbox_offsets)[BBoxAttr::H] = h_offset;
      }
    }
  }
}

template <typename Dtype>
Dtype YOLOLikeLossLayer<Dtype>::CalcIoU(
    const std::vector<Dtype>& box1, 
    const std::vector<Dtype>& box2) const {
  CHECK_EQ(box1.size(), 4);
  CHECK_EQ(box2.size(), 4);

  Dtype w = CalcOverlap(box1[BBoxAttr::X], box1[BBoxAttr::W],
                        box2[BBoxAttr::X], box2[BBoxAttr::W]);
  Dtype h = CalcOverlap(box1[BBoxAttr::Y], box1[BBoxAttr::H],
                        box2[BBoxAttr::Y], box2[BBoxAttr::H]);

  Dtype inter_area = w * h;
  Dtype union_area = (box1[BBoxAttr::W] * box1[BBoxAttr::H]) 
      + (box2[BBoxAttr::W] * box2[BBoxAttr::H]) - inter_area;
  return inter_area / union_area;
}

template <typename Dtype>
Dtype YOLOLikeLossLayer<Dtype>::CalcOverlap(
    Dtype anchor1, Dtype length1, Dtype anchor2, Dtype length2) const {
  //CHECK_GE(anchor1, 0);
  //CHECK_GT(length1, 0);
  //CHECK_GE(anchor2, 0);
  //CHECK_GT(length2, 0);

  if (bbox_anchor_ == BBoxParameter::TOP_LEFT) {
    Dtype begin = std::max(anchor1, anchor2);
    Dtype end = std::min(anchor1 + length1, anchor2 + length2);
    return (begin < end) ? end - begin : 0;
  }
  else if (bbox_anchor_ == BBoxParameter::CENTER) {
    Dtype begin = std::max(anchor1 - length1 / 2., anchor2 - length2 / 2.);
    Dtype end = std::min(anchor1 + length1 / 2., anchor2 + length2 / 2.);
    return (begin < end) ? end - begin : 0;
  }
  else {
    LOG(FATAL) << "Illegal bbox anchor";
    return 0;
  }
}

#ifdef CPU_ONLY
STUB_GPU(YOLOLikeLossLayer);
#endif

INSTANTIATE_CLASS(YOLOLikeLossLayer);
REGISTER_LAYER_CLASS(YOLOLikeLoss);

} // namespace caffe