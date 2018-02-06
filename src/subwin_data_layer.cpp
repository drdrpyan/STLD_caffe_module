#include "subwin_data_layer.hpp"

namespace caffe
{

template <typename Dtype>
void SubwinDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseImgBBoxDataLayer<Dtype>::DataLayerSetUp(bottom, top);

  const SubwinOffsetParameter& param = layer_param_.subwin_offset_param();
  
  win_size_.width = param.win_size().width();
  win_size_.height = param.win_size().height();

  win_offset_.resize(param.win_offset_size());
  for (int i = 0; i < win_offset_.size(); ++i) {
    win_offset_[i].x = param.win_offset(i).x();
    win_offset_[i].y = param.win_offset(i).y();
  }

  global_detection_ = param.global_detection();

  ReshapeTop(*(top[0]), *(top[1]), top);

  anno_encoder_.reset(new bgm::AnnoEncoder<Dtype>);
  obj_contained_.reset(new bgm::IntersectionOverObjContained<Dtype>(0.7f));
}

//template <typename Dtype>
//void SubwinDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
//                                     const vector<Blob<Dtype>*>& top) {
//  if (prefetch_current_)
//    prefetch_free_.push(prefetch_current_);
//  prefetch_current_ = prefetch_full_.pop("Waiting for data");
//
//  CHECK_EQ(prefetch_current_.data_.num(), 1);
//
//  std::vector<int> cropped_top_shape(4);
//  cropped_top_shape[0] = win_offset_.size();
//  cropped_top_shape[1] = prefetch_current_.data_.channels();
//  cropped_top_shape[2] = win_size_.height;
//  cropped_top_shape[3] = win_size_.width;
//  top[0]->Reshape(cropped_top_shape);
//
//  if (top.size() > 1) {
//    CHECK_EQ(prefetch_current_.label_.num(), 1);
//
//    if (global_detection_)
//      top[1]->ReshapeLike(prefetch_current_.label_);
//    else {
//      std::vector<int> label_shape(prefetch_current_.label_.shape());
//      label_shape[0] = win_offset_.size();
//      top[1]->Reshape(label_shape);
//    }
//  }
//
//  if (top.size() > 2)
//    top[2]->ReshapeLike(prefetch_current_.data_);
//}

template <typename Dtype>
void SubwinDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_)
    prefetch_free_.push(prefetch_current_);
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  const Blob<Dtype>& src_data = prefetch_current_->data_;
  const Blob<Dtype>& src_label = prefetch_current_->label_;

  ReshapeTop(src_data, src_label, top);

  ForwardCroppedImg_cpu(src_data, top[0]);
  
  if (top.size() > 1)
    ForwardLabelBBox(src_label, top[1], top[2]);
  if (top.size() > 3)
    ForwardWholeImg(src_data, top[3]);
}



template <typename Dtype>
void SubwinDataLayer<Dtype>::ReshapeTop(
    const Blob<Dtype>& src_data,
    const Blob<Dtype>& src_label,
    const std::vector<Blob<Dtype>*>& top) const {
  CHECK_EQ(src_data.num(), 1);

  if (top.size() > 3)
    top[3]->ReshapeLike(src_data);

  std::vector<int> cropped_top_shape(4);
  cropped_top_shape[0] = win_offset_.size();
  cropped_top_shape[1] = src_data.channels();
  cropped_top_shape[2] = win_size_.height;
  cropped_top_shape[3] = win_size_.width;
  top[0]->Reshape(cropped_top_shape);

  if (top.size() > 2) {
    CHECK_EQ(src_label.num(), 1);

    if (global_detection_) {
      std::vector<int> gt_shape(3);
      gt_shape[0] = src_label.num();
      gt_shape[1] = 1;
      gt_shape[2] = src_label.count(2);
      top[1]->Reshape(gt_shape);
      
      gt_shape[1] = 4;
      top[2]->Reshape(gt_shape);
    }
    else {
      std::vector<int> gt_shape(3);
      gt_shape[0] = win_offset_.size();
      gt_shape[1] = 1;
      gt_shape[2] = src_label.count(2);
      top[1]->Reshape(gt_shape);
      
      gt_shape[1] = 4;
      top[2]->Reshape(gt_shape);
    }
  }
}

template <typename Dtype>
void SubwinDataLayer<Dtype>::ForwardCroppedImg_cpu(
    const Blob<Dtype>& src, Blob<Dtype>* dst) const {
  const Dtype* src_ptr = src.cpu_data();
  Dtype* dst_ptr = dst->mutable_cpu_data();

  for (int i = 0; i < win_offset_.size(); ++i) {
    for (int c = 0; c < src.channels(); ++c) {
      int src_offset = src.offset(0, c, win_offset_[i].y, win_offset_[i].x);
      const Dtype* src_iter = src_ptr + src_offset;

      int dst_offset = dst->offset(i, c);
      Dtype* dst_iter = dst_ptr + dst_offset;

      for (int h = dst->height(); h--; ) {
        caffe_copy<Dtype>(dst->width(), src_iter, dst_iter);
        src_iter += src.width();
        dst_iter += dst->width();
      }
    }
  }
}


template <typename Dtype>
void SubwinDataLayer<Dtype>::ForwardLabelBBox(
    const Blob<Dtype>& src, Blob<Dtype>* label_dst, Blob<Dtype>* bbox_dst) const {
  CHECK_EQ(src.num(), 1);
  CHECK(label_dst);
  CHECK(bbox_dst);

  std::vector<std::vector<Dtype> > gt_label;
  std::vector<std::vector<cv::Rect_<Dtype> > > gt_bbox;
  BaseImgBBoxDataLayer<Dtype>::ParseLabelBBox(src, &gt_label, &gt_bbox);

  std::vector<caffe::Blob<Dtype>*> top_blobs(2);
  top_blobs[0] = label_dst;
  top_blobs[1] = bbox_dst;
  
  if (global_detection_) {    
    std::vector<std::vector<int> > gt_int_label(gt_label.size());
    for (int i = 0; i < gt_int_label.size(); ++i) {
      gt_int_label[i].resize(gt_label[i].size());
      for (int j = 0; j < gt_int_label[i].size(); ++j) {
        gt_int_label[i][j] = gt_label[i][j];
      }
    }

    anno_encoder_->Encode(gt_int_label, gt_bbox, top_blobs);
  }
  else {
    cv::Rect_<Dtype> win;
    win.width = win_size_.width;
    win.height = win_size_.height;

    std::vector<std::vector<int> > win_label(win_offset_.size());
    std::vector<std::vector<cv::Rect_<Dtype> > > win_bbox(win_offset_.size());

    for (int i = 0; i < win_offset_.size(); ++i) {
      win.x = win_offset_[i].x;
      win.y = win_offset_[i].y;

      for (int j = 0; j < gt_bbox[0].size(); ++j) {
        cv::Rect_<Dtype> bbox = gt_bbox[0][j];
        if ((*obj_contained_)(win, bbox)) {
          bbox.x -= win.x;
          bbox.y -= win.y;
          win_label[i].push_back(gt_label[0][j]);
          win_bbox[i].push_back(bbox);
        }
      }
    }

    anno_encoder_->Encode(win_label, win_bbox, top_blobs);
  }
}



//template <typename Dtype>
//void SubwinDataLayer<Dtype>::CropImg(const Dtype* src_data, Dtype* dst_data) {
//  crop_param_.set_axis(2);
//  for(int i=0; i<)
//  crop_param_.clear_offset();
//  crop_param_.add_offset()
//}

#ifdef CPU_ONLY
STUB_GPU(SubwinDataLayer);
#endif

INSTANTIATE_CLASS(SubwinDataLayer);
REGISTER_LAYER_CLASS(SubwinData);

} // namespace caffe