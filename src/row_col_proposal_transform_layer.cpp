#include "row_col_proposal_transform_layer.hpp"

namespace caffe
{

template <typename Dtype>
void RowColProposalTransformLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  RowColProposalTransformParameter param = this->layer_param_.rcp_transform_param();

  row_ = param.row();
  CHECK_GT(row_, 0);
  col_ = param.col();
  CHECK_GT(col_, 0);

  in_width_ = param.in_width();
  CHECK_GT(in_width_, 0);
  in_height_ = param.in_height();
  CHECK_GT(in_height_, 0);

  out_width_ = param.out_width();
  CHECK_GT(out_width_, 0);
  out_height_ = param.out_height();
  CHECK_GT(out_height_, 0);

  objectness_ = param.has_objectness() ? param.objectness() : false;
  CHECK(!objectness_ == param.has_num_label()) << "Specify \'objectness\' or \'num_label\'.";
  if (!objectness_) {
    num_label_ = param.num_label();
    CHECK_GT(num_label_, 0);
  }
  else
    num_label_ = 1;

  ignore_label_.clear();
  if (param.ignore_label_size() > 0) {
    for (int i = 0; i < param.ignore_label_size(); ++i)
      ignore_label_.push_back(param.ignore_label(i));
  }
}

template <typename Dtype>
void RowColProposalTransformLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& gt_label = *(bottom[0]);
  const Blob<Dtype>& gt_bbox = *(bottom[1]);

  CHECK_EQ(gt_label.num(), gt_bbox.num());
  CHECK_EQ(gt_label.height(), gt_bbox.height());
  CHECK_EQ(gt_label.channels(), 1);
  CHECK_EQ(gt_bbox.channels(), 4);

  std::vector<int> top_shape(4);
  top_shape[0] = gt_label.num();
  top_shape[1] = num_label_ * (row_ + col_);
  top_shape[2] = out_height_;
  top_shape[3] = out_width_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RowColProposalTransformLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<std::vector<int> > label;
  std::vector <std::vector<cv::Rect_<Dtype> > > bbox;
  ParseGT(*(bottom[0]), *(bottom[1]), &label, &bbox);

  int num_batch = top[0]->num();
  caffe_set(top[0]->count(), static_cast<Dtype>(0),
            top[0]->mutable_cpu_data());
  for (int n = 0; n < num_batch; ++n) {
    RCPTransform(n, label[n], bbox[n], top[0]);
  }
}

template <typename Dtype>
void RowColProposalTransformLayer<Dtype>::ParseGT(
    const Blob<Dtype>& label_blob,
    const Blob<Dtype>& bbox_blob,
    std::vector<std::vector<int> >* label,
    std::vector<std::vector<cv::Rect_<Dtype> > >* bbox) const {
  CHECK(label);
  CHECK(bbox);
  
  int num_batch = label_blob.num();

  label->resize(num_batch);
  bbox->resize(num_batch);

  const Dtype* label_ptr = label_blob.cpu_data();
  const Dtype* bbox_ptr = bbox_blob.cpu_data();
  for (int n = 0; n < num_batch; ++n) {
    (*label)[n].clear();
    (*bbox)[n].clear();

    const Dtype* label_iter = label_ptr + label_blob.offset(n, 0);
    const Dtype* x_iter = bbox_ptr + bbox_blob.offset(n, 0);
    const Dtype* y_iter = bbox_ptr + bbox_blob.offset(n, 1);
    const Dtype* w_iter = bbox_ptr + bbox_blob.offset(n, 2);
    const Dtype* h_iter = bbox_ptr + bbox_blob.offset(n, 3);

    for (int h = label_blob.height(); h--; ) {
      bool discard = (*label_iter == LabelParameter::DUMMY_LABEL);
      discard |= (*label_iter == LabelParameter::NONE);
      if (!discard && !ignore_label_.empty()) {
        auto ignore_iter = std::find(ignore_label_.cbegin(),
                                     ignore_label_.cend(),
                                     *label_iter);
        discard |= (ignore_iter != ignore_label_.cend());
      }

      if (!discard) {
        int new_label = objectness_ ? 1 : *label_iter;
        CHECK_GT(new_label, 0);
        CHECK_LE(new_label, num_label_);

        (*label)[n].push_back(new_label);
        (*bbox)[n].push_back(cv::Rect_<Dtype>(*x_iter, *y_iter,
                                              *w_iter, *h_iter));
      }

      ++label_iter;
      ++x_iter;
      ++y_iter;
      ++w_iter;
      ++h_iter;
    }
  }
}

//template <typename Dtype>
//void RowColProposalTransformLayer<Dtype>::PickGTForOutCell(
//    const std::vector<int>& all_label,
//    const std::vector<cv::Rect_<Dtype> >& all_bbox,
//    int out_r, int out_c,
//    std::vector<int>* picked_label,
//    std::vector<cv::Rect_<Dtype> >* picked_bbox) const {
//  CHECK_EQ(all_label.size(), all_bbox.size());
//  CHECK_GE(out_r, 0);
//  CHECK_LT(out_r, out_height_);
//  CHECK_GE(out_c, 0);
//  CHECK_LT(out_c, out_width_);
//  CHECK(picked_label);
//  CHECK(picked_bbox);
//
//  picked_label->clear();
//  picked_bbox->clear();
//
//  float cell_width = in_width_ / static_cast<float>(out_width_);
//  float cell_height = in_height_ / static_cast<float>(out_height_);
//  cv::Rect_<Dtype> cell(out_c * cell_width, out_r * cell_height,
//                        cell_width, cell_height);
//
//  for (int i = 0; i < all_bbox.size(); ++i) {
//    if (obj_contained_(cell, all_bbox[i])) {
//      picked_label->push_back(all_label[i]);
//
//      cv::Rect_<Dtype> new_bbox(all_bbox[i].x - cell.x,
//                                all_bbox[i].y - cell.y,
//                                all_bbox[i].width,
//                                all_bbox[i].height);
//      picked_bbox->push_back(new_bbox);
//    }
//  }
//}

template <typename Dtype>
void RowColProposalTransformLayer<Dtype>::RCPTransform(
    int batch,
    const std::vector<int>& label,
    const std::vector<cv::Rect_<Dtype> >& bbox,
    Blob<Dtype>* top) const {
  CHECK_EQ(label.size(), bbox.size());
  CHECK(top);

  Dtype row_step = in_height_ / (out_height_ * row_);
  Dtype col_step = in_width_ / (out_width_ * col_);

  Dtype* dst_ptr = top->mutable_cpu_data();

  for (int i = 0; i < label.size(); ++i) {
    Dtype center_y = bbox[i].y + (bbox[i].height / 2.);
    Dtype center_x = bbox[i].x + (bbox[i].width / 2.);

    int row_idx = GetGrid(center_y, row_step);
    CHECK_GE(row_idx, 0);
    CHECK_LT(row_idx, out_height_ * row_);
    
    int col_idx = GetGrid(center_x, col_step);
    CHECK_GE(col_idx, 0);
    CHECK_LT(col_idx, out_width_ * col_);

    int dst_c_begin = (label[i] - 1) * (row_ + col_);
    int dst_c1 = dst_c_begin + (row_idx % row_); // row index dst
    int dst_c2 = dst_c_begin + row_ + (col_idx % col_); // col index dst
    int dst_h = row_idx / row_;
    int dst_w = col_idx / col_;

    *(dst_ptr + top->offset(batch, dst_c1, dst_h, dst_w)) = 1;
    *(dst_ptr + top->offset(batch, dst_c2, dst_h, dst_w)) = 1;
  }
}

#ifdef CPU_ONLY
STUB_GPU(RowColProposalTransformLayer);
#endif

INSTANTIATE_CLASS(RowColProposalTransformLayer);
REGISTER_LAYER_CLASS(RowColProposalTransform);
} // namespace caffe