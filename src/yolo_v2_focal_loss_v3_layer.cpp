#include "yolo_v2_focal_loss_v3_layer.hpp"

#include "caffe/util/math_functions.hpp"

#define USE_YOLOV2_SIGMOID_CONF
#define USE_YOLOV2_SIGMOID_BOX_LOC

namespace caffe
{

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const YOLOV2LossParameter& param = layer_param_.yolo_v2_loss_param();

  //num_anchor_ = param.num_anchor();
  //CHECK_GT(num_anchor_, 0);

  anchor_.resize(param.anchor_size());
  for (int i = 0; i < param.anchor_size(); ++i) {
    const caffe::Rect2f& a = param.anchor(i);
    anchor_[i] = cv::Rect_<Dtype>(a.top_left().x(), a.top_left().y(),
                                  a.size().width(), a.size().height());
  }
  CHECK_GT(anchor_.size(), 0);

  num_class_ = param.num_class();
  CHECK_GT(num_class_, 0);

  img_size_.width = param.img_size().width();
  img_size_.height = param.img_size().height();
  CHECK_GT(img_size_.area(), 0);

  yolo_map_size_.width = param.yolo_map_size().width();
  yolo_map_size_.height = param.yolo_map_size().height();
  CHECK_GT(yolo_map_size_.area(), 0);

  overlap_threshold_ = param.overlap_threshold();
  CHECK_GE(overlap_threshold_, 0);
  CHECK_LE(overlap_threshold_, 1);

  noobj_scale_ = param.noobj_scale();
  CHECK_GE(noobj_scale_, 0);

  obj_scale_ = param.obj_scale();
  CHECK_GE(obj_scale_, 0);

  cls_scale_ = param.cls_scale();
  CHECK_GE(cls_scale_, 0);
  
  coord_scale_ = param.coord_scale();
  CHECK_GE(coord_scale_, 0);

  anno_decoder_.reset(new bgm::AnnoDecoder<Dtype>);

  const FocalLossParameter& focal_loss_param = layer_param_.focal_loss_param();
  focal_loss_alpha_ = focal_loss_param.alpha();
  focal_loss_gamma_ = focal_loss_param.gamma();
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(bottom[0]->channels(), 
           anchor_.size() * (NUM_ANCHOR_ELEM + num_class_));
  CHECK_EQ(bottom[0]->height(), yolo_map_size_.height);
  CHECK_EQ(bottom[0]->width(), yolo_map_size_.width);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[2]->channels(), 4);
  CHECK_EQ(bottom[1]->height(), bottom[2]->height());

  std::vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);

  diff_.ReshapeLike(*(bottom[0]));

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
  if (top.size() > 11) // max noobj
    top[11]->Reshape(loss_shape);
  if (top.size() > 12) // min obj
    top[12]->Reshape(loss_shape);
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& yolo_out = *(bottom[0]);
  Blob<Dtype>& loss_out = *(top[0]);

  std::vector<std::vector<int> > gt_label;
  std::vector<std::vector<cv::Rect_<Dtype> > > gt_bbox;
  std::vector<Blob<Dtype>*> gt_bottom(bottom.begin() + 1, bottom.end());
  anno_decoder_->Decode(gt_bottom, &gt_label, &gt_bbox); // decoder 초기화 안됐음

  Clear();

  for (int n = 0; n < yolo_out.num(); ++n) {
    for (int h = 0; h < yolo_out.height(); ++h) {
      for (int w = 0; w < yolo_out.width(); ++w) {
        //std::vector<cv::Rect_<Dtype> > shifted_anchor;
        //ShiftAnchors(h, w, &shifted_anchor);

        //int best_bbox_idx, best_anchor_idx;
        //Dtype best_iou;
        //FindBestMatching(shifted_anchor, gt_bbox[n],
        //                 &best_anchor_idx, &best_bbox_idx, &best_iou);

        //for (int a = 0; a < shifted_anchor.size(); ++a) {
        //  if ((a != best_anchor_idx) || (best_iou < overlap_threshold_))
        //    ForwardNegative(yolo_out, n, h, w, a);
        //  else {
        //    ++obj_cnt_;

        //    cv::Rect_<Dtype> true_bbox = 
        //        RawBBoxToAnchorBBox(gt_bbox[n][best_bbox_idx], 
        //                            shifted_anchor[a]);
        //    ForwardPositive(yolo_out, n, h, w, a,
        //                    gt_label[n][best_bbox_idx], true_bbox);
        //  }
        //}
        cv::Point anchor_lt = GetCellTopLeft(h, w);
        for (int a = 0; a < anchor_.size(); ++a) {
          cv::Rect_<Dtype> shifted_anchor = anchor_[a];
          shifted_anchor.x += anchor_lt.x;
          shifted_anchor.y += anchor_lt.y;

          int best_bbox_idx;
          Dtype best_iou;
          FindBestMatching(shifted_anchor, gt_bbox[n],
                           &best_bbox_idx, &best_iou);

          if (best_iou < overlap_threshold_) {
            ForwardNegative(yolo_out, n, h, w, a);
          }
          else {
            ++obj_cnt_;

            cv::Rect_<Dtype> pred_yolo_box = GetPredBBox(yolo_out, n, h, w, a);
            cv::Rect_<Dtype> pred_raw_box = YOLOBoxToRawBox(pred_yolo_box,
                                                            shifted_anchor);
            Dtype pred_iou = CalcIoU(pred_raw_box, gt_bbox[n][best_bbox_idx]);
            CHECK(!std::isnan(pred_iou));
            CHECK(!std::isinf(pred_iou));
            avg_iou_ += pred_iou;

            cv::Rect_<Dtype> true_bbox_yolo_form =
                RawBoxToYOLOBox(gt_bbox[n][best_bbox_idx], shifted_anchor);

//#ifdef USE_YOLOV2_SIGMOID_CONF
//            if (pred_iou < overlap_threshold_)
//              pred_iou = overlap_threshold_;
//#endif // USE_YOLOV2_SIGMOID_CONF

            ForwardPositive(yolo_out, n, h, w, a,
                            gt_label[n][best_bbox_idx], true_bbox_yolo_form,
                            pred_iou);
          }
        }
      }
    }
  }

  int noobj_cnt = yolo_out.num() * yolo_out.height() * yolo_out.width() - obj_cnt_;
  if (noobj_cnt > 0) {
    noobj_loss_ /= noobj_cnt;

    avg_noobj_ /= noobj_cnt;
  }

  if (obj_cnt_ > 0) {
    obj_loss_ /= obj_cnt_;
    cls_loss_ /= obj_cnt_;

    avg_obj_ /= obj_cnt_;
    avg_pos_cls_ /= obj_cnt_;
    avg_neg_cls_ /= obj_cnt_ * (num_class_ - 1);
    avg_iou_ /= obj_cnt_;
  }

  coord_loss_ /= yolo_out.num() * yolo_out.height() * yolo_out.width();
  area_loss_ /= yolo_out.num() * yolo_out.height() * yolo_out.width();

  Dtype loss = noobj_loss_ + obj_loss_ + cls_loss_ + coord_loss_ + area_loss_;
  *(loss_out.mutable_cpu_data()) = loss;

  if (top.size() > 1) // noobj loss
    top[1]->mutable_cpu_data()[0] = noobj_loss_;
  if (top.size() > 2) // obj loss
    top[2]->mutable_cpu_data()[0] = obj_loss_;
  if (top.size() > 3) // class loss
    top[3]->mutable_cpu_data()[0] = cls_loss_;
  if (top.size() > 4) // coord loss
    top[4]->mutable_cpu_data()[0] = coord_loss_;
  if (top.size() > 5) // area loss
    top[5]->mutable_cpu_data()[0] = area_loss_;
  if (top.size() > 6) // noobj conf
    top[6]->mutable_cpu_data()[0] = avg_noobj_;
  if (top.size() > 7) // obj conf
    top[7]->mutable_cpu_data()[0] = avg_obj_;
  if (top.size() > 8) // class conf
    top[8]->mutable_cpu_data()[0] = avg_neg_cls_;
  if (top.size() > 9) // pos class conf
    top[9]->mutable_cpu_data()[0] = avg_pos_cls_;
  if (top.size() > 10) // iou
    top[10]->mutable_cpu_data()[0] = avg_iou_;
  if (top.size() > 11) // max noobj
    top[11]->mutable_cpu_data()[0] = max_noobj_;
  if (top.size() > 12) // max noobj
    top[12]->mutable_cpu_data()[0] = min_obj_;
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::Clear() {
  noobj_loss_ = 0;
  obj_loss_ = 0;
  cls_loss_ = 0;
  coord_loss_ = 0;
  area_loss_ = 0;
  avg_noobj_ = 0;
  avg_obj_ = 0;
  avg_pos_cls_ = 0;
  avg_neg_cls_ = 0;
  avg_iou_ = 0;

  obj_cnt_ = 0;

  caffe_set<Dtype>(diff_.count(), static_cast<Dtype>(0), 
                   diff_.mutable_cpu_diff());

  max_noobj_ = 0;
  min_obj_ = std::numeric_limits<Dtype>::max();
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::ShiftAnchors(
    int cell_row, int cell_col,
    std::vector<cv::Rect_<Dtype> >* shifted) const {
  CHECK_GE(cell_row, 0);
  CHECK_LT(cell_row, yolo_map_size_.height);
  CHECK_GE(cell_col, 0);
  CHECK_LT(cell_col, yolo_map_size_.width);
  CHECK(shifted);

  cv::Point top_left = GetCellTopLeft(cell_row, cell_col);
  shifted->assign(anchor_.begin(), anchor_.end());
  for (auto iter = shifted->begin(); iter != shifted->end(); ++iter) {
    iter->x += top_left.x;
    iter->y += top_left.y;
  }
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::FindBestMatching(
    const std::vector<cv::Rect_<Dtype> >& anchor,
    const std::vector<cv::Rect_<Dtype> >& gt_bbox,
    int* best_anchor_idx, int* best_bbox_idx, 
    Dtype* best_iou) const {
  CHECK(best_bbox_idx);
  CHECK(best_anchor_idx);
  CHECK(best_iou);

  *best_bbox_idx = -1;
  *best_anchor_idx = -1;
  *best_iou = 0;

  for (int a = 0; a < anchor.size(); ++a) {
    for (int b = 0; b < gt_bbox.size(); ++b) {
      Dtype iou = CalcIoU(anchor[a], gt_bbox[b]);
      if (iou > *best_iou) {
        *best_iou = iou;

        *best_anchor_idx = a;
        *best_bbox_idx = b;
      }
    }
  }
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::FindBestMatching(
    const cv::Rect_<Dtype>& anchor, 
    const std::vector<cv::Rect_<Dtype> >& gt_bbox,
    int* best_bbox_idx, Dtype* best_iou) const {
  CHECK(best_bbox_idx);
  CHECK(best_iou);

  *best_bbox_idx = -1;
  *best_iou = 0;
  for (int i = 0; i < gt_bbox.size(); ++i) {
    Dtype iou = CalcIoU(anchor, gt_bbox[i]);

    if (iou > *best_iou) {
      *best_iou = iou;
      *best_bbox_idx = i;
    }
  }
}

template <typename Dtype>
cv::Rect_<Dtype> YOLOV2FocalLossV3Layer<Dtype>::RawBBoxToAnchorBBox(
    const cv::Rect_<Dtype>& raw_bbox, const cv::Rect_<Dtype>& anchor) const {
  Dtype x = (raw_bbox.x + (raw_bbox.width / 2.) - anchor.x) / anchor.width;
  Dtype y = (raw_bbox.y + (raw_bbox.height / 2.) - anchor.y) / anchor.height;
  Dtype w = std::log(raw_bbox.width / anchor.width);
  Dtype h = std::log(raw_bbox.height / anchor.height);
  return cv::Rect_<Dtype>(x, y, w, h);
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::ForwardNegative(
    const Blob<Dtype>& input, int n, int h, int w, int anchor) {
  // conf
  ForwardConf(input, n, h, w, anchor, noobj_scale_, 0, 
              noobj_loss_, avg_noobj_);

  // bbox
  ForwardBBox(input, n, h, w, anchor, noobj_scale_ * coord_scale_);
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::ForwardPositive(
    const Blob<Dtype>& input, int n, int h, int w, int anchor,
    Dtype true_label, const cv::Rect_<Dtype>& true_bbox_yolo_form,
    Dtype iou) {
  // conf
  ForwardConf(input, n, h, w, anchor, obj_scale_, 
              Sigmoid(iou) + 1 - Sigmoid(1), obj_loss_, avg_obj_);

  // bbox
  ForwardBBox(input, n, h, w, anchor, obj_scale_ * coord_scale_,
              true_bbox_yolo_form);

  // class
  ForwardClass(input, n, h, w, anchor, true_label);
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::ForwardConf(
    const Blob<Dtype>& input, 
    int n, int h, int w, int anchor, Dtype scale, Dtype iou,
    Dtype& loss, Dtype& sum_conf) {
  const Dtype* input_data = input.cpu_data();
  Dtype* diff_data = diff_.mutable_cpu_diff();

  int conf_offset = 
      input.offset(n, GetAnchorChannel(anchor, AnchorChannel::CONF), h, w);
  Dtype conf = *(input_data + conf_offset);

#ifdef USE_YOLOV2_SIGMOID_CONF
  Dtype sig_conf = Sigmoid(conf);
  sum_conf += sig_conf;

  Dtype focal_loss, focal_loss_diff;
  focal_loss_.SigmoidRegressionFocalLoss(conf, iou,
                                         focal_loss_alpha_, focal_loss_gamma_,
                                         &focal_loss, &focal_loss_diff);
  //CHECK(!std::isnan(focal_loss_diff));
  //CHECK(!std::isinf(focal_loss_diff));

  loss += scale * focal_loss;
  *(diff_data + conf_offset) = scale * focal_loss_diff;

  if (&loss == &noobj_loss_) {
    if (max_noobj_ < sig_conf)
      max_noobj_ = sig_conf;
  }
  else if (&loss == &obj_loss_) {
    if (min_obj_ > sig_conf)
      min_obj_ = sig_conf;
  }
#else
  loss += scale * std::pow(conf - iou, 2);
  *(diff_data + conf_offset) = scale * (conf - iou);
  sum_conf += conf;
#endif // USE_YOLOV2_SIGMOID_CONF
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::ForwardBBox(
    const Blob<Dtype>& input, int n, int h, int w, int anchor,
    Dtype scale, const cv::Rect_<Dtype>& target_yolo_form) {
  const Dtype* input_data = input.cpu_data();
  Dtype* diff_data = diff_.mutable_cpu_diff();

  int x_offset =
      input.offset(n, GetAnchorChannel(anchor, AnchorChannel::X), h, w);
  int y_offset =
      input.offset(n, GetAnchorChannel(anchor, AnchorChannel::Y), h, w);
  int w_offset =
      input.offset(n, GetAnchorChannel(anchor, AnchorChannel::W), h, w);
  int h_offset =
      input.offset(n, GetAnchorChannel(anchor, AnchorChannel::H), h, w);

  Dtype x = *(input_data + x_offset);
  Dtype y = *(input_data + y_offset);
  Dtype width = *(input_data + w_offset);
  Dtype height = *(input_data + h_offset);

#ifdef USE_YOLOV2_SIGMOID_BOX_LOC
  Dtype sig_x = Sigmoid(x);
  Dtype sig_y = Sigmoid(y);
  coord_loss_ += scale * std::pow(sig_x - target_yolo_form.x, 2.);
  coord_loss_ += scale * std::pow(sig_y - target_yolo_form.y, 2.);
  *(diff_data + x_offset) = scale * (sig_x - target_yolo_form.x) * sig_x * (1 - sig_x);
  *(diff_data + y_offset) = scale * (sig_y - target_yolo_form.y) * sig_y * (1 - sig_y);
#else
  coord_loss_ += coord_scale_ * std::pow(x - target_yolo_form.x, 2);
  coord_loss_ += coord_scale_ * std::pow(y - target_yolo_form.y, 2);  
  *(diff_data + x_offset) = coord_scale_ * (sig_x - target_yolo_form.x);
  *(diff_data + y_offset) = coord_scale_ * (sig_y - target_yolo_form.y);
#endif // USE_YOLOV2_SIGMOID_BOX_LOC

  area_loss_ += scale * std::pow(width - target_yolo_form.width, 2);
  area_loss_ += scale * std::pow(height - target_yolo_form.height, 2);
  *(diff_data + w_offset) = scale * (width - target_yolo_form.width);
  *(diff_data + h_offset) = scale * (height - target_yolo_form.height);
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::ForwardClass(
    const Blob<Dtype>& input, int n, int h, int w, int anchor, int true_label) {
  CHECK_GT(true_label, 0);
  CHECK_LE(true_label, num_class_);

  const Dtype* input_data = input.cpu_data();
  Dtype* diff_data = diff_.mutable_cpu_diff();

  std::vector<int> cls_offset(num_class_);
  for(int l=1; l<=num_class_; ++l)
    cls_offset[l-1] = input.offset(n, GetClassChannel(anchor, l), h, w);

  std::vector<Dtype> cls_conf(num_class_);
  for (int l = 0; l < num_class_; ++l)
    cls_conf[l] = *(input_data + cls_offset[l]);

  // softmax
  std::vector<Dtype> softmax(num_class_);
  Dtype exp_sum = 0;
  for (int l = 0; l < num_class_; ++l) {
    softmax[l] = std::exp(cls_conf[l]);
    exp_sum += softmax[l];
  }
  for (int l = 0; l < num_class_; ++l) {
    if (l != (true_label - 1))
      avg_neg_cls_ += softmax[l] / exp_sum;
    else
      avg_pos_cls_ += softmax[l] / exp_sum;
  }

  Dtype cls_loss;
  std::vector<Dtype> cls_diff;
  focal_loss_.SoftmaxFocalLoss(cls_conf, true_label-1,
                               focal_loss_alpha_, focal_loss_gamma_,
                               &cls_loss, &cls_diff);

  cls_loss_ += cls_scale_ * cls_loss;
  for (int l = 0; l < num_class_; ++l) {
    CHECK(!std::isnan(cls_diff[l]));
    CHECK(!std::isinf(cls_diff[l]));
    *(diff_data + cls_offset[l]) = cls_scale_ * cls_diff[l];
  }
}

template <typename Dtype>
void YOLOV2FocalLossV3Layer<Dtype>::Backward_cpu(
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
    caffe_cpu_axpby(bottom[0]->count(), alpha, diff_.cpu_diff(),
                    Dtype(0), bottom[0]->mutable_cpu_diff());
  }
}

template <typename Dtype>
cv::Point YOLOV2FocalLossV3Layer<Dtype>::GetCellTopLeft(int row, int col) const {
  CHECK_GE(row, 0);
  CHECK_LT(row, yolo_map_size_.height);
  CHECK_GE(col, 0);
  CHECK_LT(col, yolo_map_size_.width);

  int x = (img_size_.width / yolo_map_size_.width) * col;
  int y = (img_size_.height / yolo_map_size_.height) * row;

  return cv::Point(x, y);
}

template <typename Dtype>
Dtype YOLOV2FocalLossV3Layer<Dtype>::CalcIoU(
    const cv::Rect_<Dtype>& box1, const cv::Rect_<Dtype>& box2) const {
  Dtype h_overlap = CalcOverlap(box1.x, box1.width, box2.x, box2.width);
  Dtype v_overlap = CalcOverlap(box1.y, box1.height, box2.y, box2.height);
  Dtype intersection = h_overlap * v_overlap;
  Dtype box_union = box1.area() + box2.area() - intersection;
  return intersection / box_union;
}

template <typename Dtype>
Dtype YOLOV2FocalLossV3Layer<Dtype>::CalcOverlap(
    Dtype anchor1, Dtype length1, Dtype anchor2, Dtype length2) const {
  Dtype begin = std::max(anchor1, anchor2);
  Dtype end = std::min(anchor1 + length1, anchor2 + length2);
  return (begin < end) ? end - begin : 0;
}

template <typename Dtype>
cv::Rect_<Dtype> YOLOV2FocalLossV3Layer<Dtype>::GetPredBBox(
    const Blob<Dtype>& input, int n, int h, int w, int anchor) const {
  const Dtype* input_data = input.cpu_data();
  Dtype x = *(input_data + input.offset(n, GetAnchorChannel(anchor, AnchorChannel::X), h, w));
  Dtype y = *(input_data + input.offset(n, GetAnchorChannel(anchor, AnchorChannel::Y), h, w));
  Dtype width = *(input_data + input.offset(n, GetAnchorChannel(anchor, AnchorChannel::W), h, w));
  Dtype height = *(input_data + input.offset(n, GetAnchorChannel(anchor, AnchorChannel::H), h, w));
  return cv::Rect_<Dtype>(x, y, width, height);
}

template <typename Dtype>
cv::Rect_<Dtype> YOLOV2FocalLossV3Layer<Dtype>::RawBoxToYOLOBox(
    const cv::Rect_<Dtype>& raw_box, const cv::Rect_<Dtype>& anchor) const {
  cv::Rect_<Dtype> yolo_box;
  yolo_box.x = (raw_box.x + (raw_box.width / 2.) - anchor.x) / anchor.width;
  yolo_box.y = (raw_box.y + (raw_box.height / 2.) - anchor.y) / anchor.height;
  yolo_box.width = std::log(raw_box.width / anchor.width);
  yolo_box.height = std::log(raw_box.height / anchor.height);

  return yolo_box;
}

template <typename Dtype>
cv::Rect_<Dtype> YOLOV2FocalLossV3Layer<Dtype>::YOLOBoxToRawBox(
    const cv::Rect_<Dtype>& yolo_box, const cv::Rect_<Dtype>& anchor,
    bool shift) const {
  cv::Rect_<Dtype> raw_box;
  raw_box.width = std::exp(yolo_box.width) * anchor.width;
  raw_box.height = std::exp(yolo_box.height) * anchor.height;
  raw_box.x = (Sigmoid(yolo_box.x) * anchor.width) - (raw_box.width / 2.);
  raw_box.y = (Sigmoid(yolo_box.y) * anchor.height) - (raw_box.height / 2.);
  if (shift) {
    raw_box.x += anchor.x;
    raw_box.y += anchor.y;
  }

  return raw_box;
}

#ifdef CPU_ONLY
STUB_GPU(YOLOV2FocalLossV3Layer);
#endif

INSTANTIATE_CLASS(YOLOV2FocalLossV3Layer);
REGISTER_LAYER_CLASS(YOLOV2FocalLossV3);

} // namespace caffe