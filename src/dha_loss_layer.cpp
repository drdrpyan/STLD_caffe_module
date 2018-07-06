#include "dha_loss_layer.hpp"

namespace caffe
{

template <typename Dtype>
void DHALossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				  			                     const vector<Blob<Dtype>*>& top) {
  const DHALossParameter& param = layer_param_.dha_loss_param();
  
  anchor_.resize(param.anchor_size());
  for (int i = 0; i < param.anchor_size(); ++i) {
    const caffe::Rect2f& a = param.anchor(i);
    anchor_[i] = cv::Rect_<Dtype>(a.top_left().x(), a.top_left().y(),
                                  a.size().width(), a.size().height());
  }
  CHECK_GT(anchor_.size(), 0);

  num_class_ = param.num_class();
  CHECK_GT(num_class_, 0);

  cell_size_.width = param.cell_size().width();
  cell_size_.height = param.cell_size().height();
  CHECK_GT(cell_size_.area(), 0);

  //overlap_threshold_ = param.overlap_threshold();
  //CHECK_GE(overlap_threshold_, 0);
  //CHECK_LE(overlap_threshold_, 1);

  noobj_scale_ = param.noobj_scale();
  CHECK_GE(noobj_scale_, 0);

  obj_scale_ = param.obj_scale();
  CHECK_GE(obj_scale_, 0);

  cls_scale_ = param.cls_scale();
  CHECK_GE(cls_scale_, 0);
  
  coord_scale_ = param.coord_scale();
  CHECK_GE(coord_scale_, 0);

  anchor_wfl_alpha_.resize(param.pos_anchor_wfl().alpha_size() + 1);
  anchor_wfl_gamma_.resize(param.pos_anchor_wfl().gamma_size() + 1);
  CHECK_GT(anchor_wfl_alpha_.size(), 1);
  CHECK_EQ(anchor_wfl_alpha_.size(), anchor_wfl_gamma_.size());
  neg_anchor_id_ = anchor_wfl_alpha_.size() - 1;
  for (int i = 0; i < neg_anchor_id_; ++i) {
    anchor_wfl_alpha_[i] = param.pos_anchor_wfl().alpha(i);
    anchor_wfl_gamma_[i] = param.pos_anchor_wfl().gamma(i);
  }
  anchor_wfl_alpha_[neg_anchor_id_] = param.neg_anchor_fl().alpha();
  anchor_wfl_gamma_[neg_anchor_id_] = param.neg_anchor_fl().gamma();

  class_wfl_alpha_.resize(num_class_);
  class_wfl_gamma_.resize(num_class_);
  CHECK_EQ(param.class_wfl().alpha_size(), num_class_);
  CHECK_EQ(param.class_wfl().gamma_size(), num_class_);
  for (int i = 0; i < num_class_; ++i) {
    class_wfl_alpha_[i] = param.class_wfl().alpha(i);
    class_wfl_gamma_[i] = param.class_wfl().gamma(i);
  }

  anno_decoder_.reset(new bgm::AnnoDecoder<Dtype>);
}

template <typename Dtype>
void DHALossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  int num_channels = (anchor_.size() + 1) + (anchor_.size() * (4 + num_class_));
  CHECK_EQ(bottom[0]->channels(), num_channels);

  diff_.ReshapeLike(*(bottom[0]));

  if (bottom[0]->height() != map_size_.height ||
      bottom[0]->width() != map_size_.width) {
    CHECK_GT(bottom[0]->height(), 0);
    CHECK_GT(bottom[0]->width(), 0);
    map_size_.height = bottom[0]->height();
    map_size_.width = bottom[0]->width();
    num_cell_ = map_size_.area();
    InitAnchorMap();
  }

  std::vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);

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
  if (top.size() > 8) // neg class conf
    top[8]->Reshape(loss_shape);
  if (top.size() > 9) // pos class conf
    top[9]->Reshape(loss_shape);
  //if (top.size() > 10) // iou
  //  top[10]->Reshape(loss_shape);
  if (top.size() > 10) // anchor cls loss
    top[10]->Reshape(loss_shape);
  if (top.size() > 11) // anchor pos cls conf
    top[11]->Reshape(loss_shape);
  if (top.size() > 12) // anchor neg cls conf
    top[12]->Reshape(loss_shape);
}

template <typename Dtype>
void DHALossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  Clear();

  const Blob<Dtype>& detection_out = *(bottom[0]);
  Blob<Dtype>& loss_out = *(top[0]);

  std::vector<std::vector<int> > gt_label;
  std::vector<std::vector<cv::Rect_<Dtype> > > gt_bbox;
  std::vector<Blob<Dtype>*> gt_bottom(bottom.begin() + 1, bottom.end());
  anno_decoder_->Decode(gt_bottom, &gt_label, &gt_bbox);

  for (int n = 0; n < detection_out.num(); ++n) {
    //if (gt_bbox[n].size() > 1)
    //  Clear();
    std::vector<std::pair<int, int> > best_match;
    FindBestMatch(gt_bbox[n], &best_match);

    std::vector<int> best_match_idx;
    ArgSortByCellIdx(best_match, &best_match_idx);

    auto match_idx_iter = best_match_idx.cbegin();
    int gt_idx = -1;
    int pos_cell_idx = -1;
    if (match_idx_iter != best_match_idx.cend()) {
      gt_idx = *match_idx_iter;
      pos_cell_idx = best_match[gt_idx].first;
    }
    else {
      pos_cell_idx = -1;
    }

    for (int r = 0; r < map_size_.height; ++r) {
      for (int c = 0; c < map_size_.width; ++c) {
        int cell_idx = GetCellIdx(r, c);
        if (cell_idx == pos_cell_idx) {
          ForwardPositive(detection_out, n, r, c,
                          best_match[gt_idx].second,
                          gt_bbox[n][gt_idx], gt_label[n][gt_idx]);
          ++match_idx_iter;
          if (match_idx_iter != best_match_idx.cend()) {
            gt_idx = *match_idx_iter;
            pos_cell_idx = best_match[gt_idx].first;
          }
          else {
            pos_cell_idx = -1;
          }
        }
        else
          ForwardNegative(detection_out, n, r, c);
      }
    }
  }

  int noobj_cnt = detection_out.num() * detection_out.height() * detection_out.width() - obj_cnt_;
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

    anchor_cls_loss_ /= obj_cnt_;
    avg_pos_anchor_ /= obj_cnt_;
    avg_neg_anchor_ /= obj_cnt_ * (anchor_.size() - 1);
  }

  coord_loss_ /= detection_out.num() * detection_out.height() * detection_out.width();
  area_loss_ /= detection_out.num() * detection_out.height() * detection_out.width();

  Dtype loss = noobj_loss_ + obj_loss_ + cls_loss_ + coord_loss_ + area_loss_ + anchor_cls_loss_;
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
  //if (top.size() > 10) // iou
  //  top[10]->mutable_cpu_data()[0] = avg_iou_;
  if (top.size() > 10) // anchor cls loss
    top[10]->mutable_cpu_data()[0] = anchor_cls_loss_;
  if (top.size() > 11) // anchor pos cls conf
    top[11]->mutable_cpu_data()[0] = avg_pos_anchor_;
  if (top.size() > 12) // anchor neg cls conf
    top[12]->mutable_cpu_data()[0] = avg_neg_anchor_;
}

template <typename Dtype>
void DHALossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
void DHALossLayer<Dtype>::InitAnchorMap() {
  anchor_map_.resize(map_size_.area(), anchor_);

  int y_offset = 0;
  for (int r = 0; r < map_size_.height; ++r) {
    int x_offset = 0;
    for (int c = 0; c < map_size_.width; ++c) {
      for (int a = 0; a < anchor_.size(); ++a) {
        std::vector<cv::Rect_<Dtype> >& current_cell = 
          anchor_map_[r * map_size_.width + c];
        current_cell[a].x += x_offset;
        current_cell[a].y += y_offset;
      }
      x_offset += cell_size_.width;
    }
    y_offset += cell_size_.height;
  }
}

template <typename Dtype>
void DHALossLayer<Dtype>::Clear() {
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

  anchor_cls_loss_ = 0;
  avg_pos_anchor_ = 0;
  avg_neg_anchor_ = 0;

  obj_cnt_ = 0;

  caffe_set<Dtype>(diff_.count(), static_cast<Dtype>(0), 
                   diff_.mutable_cpu_diff());

  //max_noobj_ = 0;
  //min_obj_ = std::numeric_limits<Dtype>::max();
}


////template <typename Dtype>
////void DHALossLayer<Dtype>::InitAnchorMap() {
////  anchor_map_.resize(map_size_.area() * anchor_.size());
////
////  auto map_iter = anchor_map_.begin();
////  int y_offset = 0;
////  for (int r = 0; r < map_size_.height; ++r) {
////    int x_offset = 0;
////    for (int c = 0; c < map_size_.width; ++c) {
////      for (int a = 0; a < anchor_.size(); ++a) {
////        const cv::Rect_<Dtype>& anchor = anchor_[a];
////        map_iter->x = anchor.x + x_offset;
////        map_iter->y = anchor.y + y_offset;
////        map_iter->width = anchor.width;
////        map_iter->height = anchor.height;
////        ++map_iter;
////      }
////      x_offset += cell_size_.width;
////    }
////    y_offset += cell_size_.height;
////  }
////}
//
////template <typename Dtype>
////void DHALossLayer<Dtype>::FindBestMatching(
////    const std::vector<cv::Rect_<Dtype> >& true_gt_box,
////    std::vector<Matching>* best_matching) const {
////  CHECK(best_matching);
////  best_matching->resize(true_gt_box.size());
////  for (int i = 0; i < best_matching->size(); ++i)
////    (*best_matching)[i].gt_idx = i;
////
////  if (true_gt_box.empty()) return;
////
////  //CHECK_GT(map_rows, 0);
////  //CHECK_GT(map_cols, 0);
////  //CHECK_EQ(map_rows * map_cols, anchor_map_.size());
////
////  std::vector<Dtype> best_iou(true_gt_box.size(), -1);
////
////  for (int i = 0; i < anchor_map_.size(); ++i) {
////    for (int j = 0; j < anchor_.size(); ++j) {
////      for (int k = 0; k < true_gt_box.size(); ++k) {
////        Dtype iou = CalcIoU(anchor_map_[i][j], true_gt_box[g]);
////        if (iou > best_iou[k]) {
////          best_iou[k] = iou;
////          (*best_matching)[k].cell_idx = i;
////          (*best_matching)[k].anchor_idx = j;
////        }
////      }
////    }
////  }
////}

template <typename Dtype>
void DHALossLayer<Dtype>::FindBestMatch(
    const std::vector<cv::Rect_<Dtype> >& true_gt_box,
    std::vector<std::pair<int, int> >* best_match) const {
  CHECK(best_match);
  best_match->resize(true_gt_box.size());

  if (best_match->size() > 0) {
    std::vector<Dtype> box_area(true_gt_box.size());
    for (int i = 0; i < box_area.size(); ++i)
      box_area[i] = true_gt_box[i].area();
    std::vector<int> box_idx(true_gt_box.size());
    std::iota(box_idx.begin(), box_idx.end(), 0);
    std::sort(box_idx.begin(), box_idx.end(),
              [&box_area](int i1, int i2) {
      return (box_area[i1] < box_area[i2]); });

    std::vector<bool> cell_matched(anchor_map_.size(), false);

    for (int i = 0; i < box_idx.size(); ++i) {
      const cv::Rect_<Dtype>& box = true_gt_box[box_idx[i]];
      int best_cell_idx = -1;
      int best_anchor_idx = -1;
      Dtype best_iou = -1;

      for (int j = 0; j < anchor_map_.size(); ++j) {
        if (!cell_matched[j]) {
          const std::vector<cv::Rect_<Dtype> >& cell_anchor = anchor_map_[j];
          int temp_anchor_idx = -1;
          Dtype temp_best_iou = -1;

          for (int k = 0; k < anchor_.size(); ++k) {
            Dtype iou = CalcIoU(box, cell_anchor[k]);
            if (iou > temp_best_iou) {
              temp_best_iou = iou;
              temp_anchor_idx = k;
            }
          }

          if (temp_best_iou > best_iou) {
            best_iou = temp_best_iou;
            best_cell_idx = j;
            best_anchor_idx = temp_anchor_idx;
          }
        }
      }

      cell_matched[best_cell_idx] = true;

      std::pair<int, int>& current_match = (*best_match)[box_idx[i]];
      current_match.first = best_cell_idx;
      current_match.second = best_anchor_idx;
    }
  }
}

template <typename Dtype>
void DHALossLayer<Dtype>::ArgSortByCellIdx(
    const std::vector<std::pair<int, int> >& match,
    std::vector<int>* sorted_idx) const {
  CHECK(sorted_idx);
  
  sorted_idx->resize(match.size());
  std::iota(sorted_idx->begin(), sorted_idx->end(), 0);

  std::sort(sorted_idx->begin(), sorted_idx->end(),
              [&match](int i1, int i2) {
      return (match[i1].first < match[i2].first); });
}

template <typename Dtype>
void DHALossLayer<Dtype>::ForwardPositive(const Blob<Dtype>& input,
                                          int n, int cell_r, int cell_c,
                                          int true_anchor, 
                                          const cv::Rect_<Dtype>& bbox,
                                          int true_label) {
  //ForwardAnchorScore(input, n, cell_r, cell_c, true_anchor);
  ForwardNegAnchorScore(input, n, cell_r, cell_c, true);
  ForwardPosAnchorScore(input, n, cell_r, cell_c, true_anchor);

  for (int i = 0; i < anchor_.size(); ++i) {
    if (i != true_anchor)
      ForwardBBox(input, n, cell_r, cell_c, i);
    else {
      int cell_idx = GetCellIdx(cell_r, cell_c);
      ForwardBBox(input, n, cell_r, cell_c, i,
                  RawBoxToAnchorRelativeForm(bbox, 
                                             anchor_map_[cell_idx][i]));
    }
  }
  ForwardClassScore(input, n, cell_r, cell_c, true_anchor,
                    true_label - 1, true);
}

template <typename Dtype>
void DHALossLayer<Dtype>::ForwardNegative(const Blob<Dtype>& input,
                                          int n, int cell_r, int cell_c) {
  //ForwardAnchorScore(input, n, cell_r, cell_c, neg_anchor_id_);
  ForwardNegAnchorScore(input, n, cell_r, cell_c, false);
  for (int i = 0; i < anchor_.size(); ++i)
    ForwardBBox(input, n, cell_r, cell_c, i);
}

template <typename Dtype>
void DHALossLayer<Dtype>::ForwardNegAnchorScore(
    const Blob<Dtype>& input, int n, int cell_r, int cell_c, bool target) {
  const Dtype* input_data = input.cpu_data();

  int offset = input.offset(n, GetAnchorScoreChannel(neg_anchor_id_),
                            cell_r, cell_c);
  Dtype score = input_data[offset];
  Dtype sig_score = Sigmoid(score);

  Dtype loss, diff;
  focal_loss_.SigmoidRegressionFocalLoss(score, target ? 1 : 0,
                                         anchor_wfl_alpha_[neg_anchor_id_],
                                         anchor_wfl_gamma_[neg_anchor_id_],
                                         &loss, &diff);

  Dtype* diff_data = diff_.mutable_cpu_diff();
  diff_data[offset] = target ? (obj_scale_ * diff) : (noobj_scale_ * diff);

  if (target) {
    obj_loss_ += obj_scale_ * loss;
    avg_obj_ += sig_score;
    ++obj_cnt_;
  }
  else {
    noobj_loss_ += noobj_scale_ * loss;
    avg_noobj_ += sig_score;
  }
}

template <typename Dtype>
void DHALossLayer<Dtype>::ForwardPosAnchorScore(
    const Blob<Dtype>& input, int n, int cell_r, int cell_c, int true_anchor) {
  const Dtype* input_data = input.cpu_data();

  std::vector<int> anchor_score_offset(anchor_.size());
  std::vector<Dtype> anchor_score(anchor_.size());

  for (int i = 0; i < anchor_score.size(); ++i) {
    anchor_score_offset[i] = input.offset(n, GetAnchorScoreChannel(i),
                                          cell_r, cell_c);
    anchor_score[i] = input_data[anchor_score_offset[i]];
  }

  Dtype loss;
  std::vector<Dtype> softmax, softmax_diff;
  focal_loss_.SoftmaxFocalLoss(anchor_score, true_anchor,
                               anchor_wfl_alpha_[true_anchor],
                               anchor_wfl_gamma_[true_anchor],
                               &softmax, &loss, &softmax_diff);

  Dtype* diff_data = diff_.mutable_cpu_diff();
  for (int i = 0; i < anchor_score_offset.size(); ++i)
    diff_data[anchor_score_offset[i]] = softmax_diff[i];

  anchor_cls_loss_ += loss;
  for (int i = 0; i < anchor_score.size(); ++i) {
    if (i != true_anchor)
      avg_neg_anchor_ += softmax[i];
    else
      avg_pos_anchor_ += softmax[true_anchor];
  }
}

template <typename Dtype>
void DHALossLayer<Dtype>::ForwardAnchorScore(
    const Blob<Dtype>& input, int n, 
    int cell_r, int cell_c, int true_anchor) {
  const Dtype* input_data = input.cpu_data();
  std::vector<int> anchor_score_offset(anchor_.size() + 1);
  std::vector<Dtype> anchor_score(anchor_.size() + 1);

  for (int i = 0; i < anchor_score.size(); ++i) {
    anchor_score_offset[i] = input.offset(n, GetAnchorScoreChannel(i),
                                          cell_r, cell_c);
    anchor_score[i] = input_data[anchor_score_offset[i]];
  }

  Dtype loss;
  std::vector<Dtype> softmax, softmax_diff;
  focal_loss_.SoftmaxFocalLoss(anchor_score, true_anchor,
                               anchor_wfl_alpha_[true_anchor],
                               anchor_wfl_gamma_[true_anchor],
                               &softmax, &loss, &softmax_diff);

  Dtype* diff_data = diff_.mutable_cpu_diff();
  for (int i = 0; i < anchor_score_offset.size(); ++i)
    diff_data[anchor_score_offset[i]] = softmax_diff[i];

  if (true_anchor == neg_anchor_id_) {
    noobj_loss_ += noobj_scale_ * loss;
    avg_noobj_ += softmax[neg_anchor_id_];
  }
  else {
    obj_loss_ += obj_scale_ * loss;
    avg_obj_ += softmax[true_anchor];
    //avg_obj_ += (1 - softmax[neg_anchor_id_]);
    ++obj_cnt_;
  }
}

template <typename Dtype>
void DHALossLayer<Dtype>::ForwardBBox(
    const Blob<Dtype>& input, int n, int cell_r, int cell_c, int anchor_id,
    const cv::Rect_<Dtype>& target_box_form) {
  CHECK_GE(anchor_id, 0);
  CHECK_LE(anchor_id, anchor_.size());

  const Dtype* input_data = input.cpu_data();
  Dtype* diff_data = diff_.mutable_cpu_diff();

  int x_offset =
      input.offset(n, GetAnchorElemChannel(anchor_id, AnchorElemID::X), 
                   cell_r, cell_c);
  int y_offset =
      input.offset(n, GetAnchorElemChannel(anchor_id, AnchorElemID::Y),
                   cell_r, cell_c);
  int w_offset =
      input.offset(n, GetAnchorElemChannel(anchor_id, AnchorElemID::W), 
                   cell_r, cell_c);
  int h_offset =
      input.offset(n, GetAnchorElemChannel(anchor_id, AnchorElemID::H), 
                   cell_r, cell_c);

  Dtype x = *(input_data + x_offset);
  Dtype y = *(input_data + y_offset);
  Dtype width = *(input_data + w_offset);
  Dtype height = *(input_data + h_offset);

  Dtype sig_x = Sigmoid(x);
  Dtype sig_y = Sigmoid(y);
  coord_loss_ += coord_scale_ * std::pow(sig_x - target_box_form.x, 2.);
  coord_loss_ += coord_scale_ * std::pow(sig_y - target_box_form.y, 2.);
  *(diff_data + x_offset) = coord_scale_ * (sig_x - target_box_form.x) * sig_x * (1 - sig_x);
  *(diff_data + y_offset) = coord_scale_ * (sig_y - target_box_form.y) * sig_y * (1 - sig_y);

  area_loss_ += coord_scale_ * std::pow(width - target_box_form.width, 2);
  area_loss_ += coord_scale_ * std::pow(height - target_box_form.height, 2);
  *(diff_data + w_offset) = coord_scale_ * (width - target_box_form.width);
  *(diff_data + h_offset) = coord_scale_ * (height - target_box_form.height);
}

template <typename Dtype>
void DHALossLayer<Dtype>::ForwardClassScore(const Blob<Dtype>& input,
                                            int n, int cell_r, int cell_c,
                                            int anchor_id, int true_class,
                                            bool class_id_zero_begin) {
  const int true_cls_idx = class_id_zero_begin ? true_class : true_class - 1;
  
  CHECK_GE(true_cls_idx, 0);
  CHECK_LT(true_cls_idx, num_class_);

  const Dtype* input_data = input.cpu_data();
  std::vector<int> class_score_offset(num_class_);
  std::vector<Dtype> class_score(num_class_);

  for (int i = 0; i < class_score.size(); ++i) {
    class_score_offset[i] = input.offset(
        n, GetClassScoreChannel(anchor_id, i, true),
        cell_r, cell_c);
    class_score[i] = input_data[class_score_offset[i]];
  }

  Dtype loss;
  std::vector<Dtype> softmax, softmax_diff;
  focal_loss_.SoftmaxFocalLoss(class_score, true_cls_idx,
                               class_wfl_alpha_[true_cls_idx],
                               class_wfl_gamma_[true_cls_idx],
                               &softmax, &loss, &softmax_diff);

  cls_loss_ += cls_scale_ * loss;

  Dtype* diff_data = diff_.mutable_cpu_diff();
  for (int i = 0; i < class_score_offset.size(); ++i)
    diff_data[class_score_offset[i]] = cls_scale_ * softmax_diff[i];

  for (int i = 0; i < num_class_; ++i) {
    if (i != true_cls_idx)
      avg_neg_cls_ += softmax[i];
    else
      avg_pos_cls_ += softmax[i];
  }
}

template <typename Dtype>
Dtype DHALossLayer<Dtype>::CalcIoU(
    const cv::Rect_<Dtype>& box1, const cv::Rect_<Dtype>& box2) const {
  Dtype h_overlap = CalcOverlap(box1.x, box1.width, box2.x, box2.width);
  Dtype v_overlap = CalcOverlap(box1.y, box1.height, box2.y, box2.height);
  Dtype intersection = h_overlap * v_overlap;
  Dtype box_union = box1.area() + box2.area() - intersection;
  return intersection / box_union;
}

template <typename Dtype>
Dtype DHALossLayer<Dtype>::CalcOverlap(
    Dtype anchor1, Dtype length1, Dtype anchor2, Dtype length2) const {
  Dtype begin = std::max(anchor1, anchor2);
  Dtype end = std::min(anchor1 + length1, anchor2 + length2);
  return (begin < end) ? end - begin : 0;
}

template <typename Dtype>
cv::Rect_<Dtype> DHALossLayer<Dtype>::RawBoxToAnchorRelativeForm(
    const cv::Rect_<Dtype>& raw_box, const cv::Rect_<Dtype>& anchor) const {
  cv::Rect_<Dtype> relative_form;
  relative_form.x = (raw_box.x + (raw_box.width / 2.) - anchor.x) / anchor.width;
  relative_form.y = (raw_box.y + (raw_box.height / 2.) - anchor.y) / anchor.height;
  relative_form.width = std::log(raw_box.width / anchor.width);
  relative_form.height = std::log(raw_box.height / anchor.height);

  return relative_form;
}

#ifdef CPU_ONLY
STUB_GPU(DHALossLayer);
#endif

INSTANTIATE_CLASS(DHALossLayer);
REGISTER_LAYER_CLASS(DHALoss);
} // namespace caffe