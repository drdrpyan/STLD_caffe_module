#include "minibatch_data_layer.hpp"

#include <random>
#include <chrono>

#ifndef NDEBUG
#include <opencv2/core.hpp>
#endif // !NDEBUG

namespace caffe
{

template <typename Dtype>
void MinibatchDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BaseImgBBoxDataLayer<Dtype>::DataLayerSetUp(bottom, top);

  std::uniform_int_distribution<int> dist;
  rng_ = std::bind(dist,
                   std::default_random_engine(
                      std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count()));

  obj_contained_.reset(new bgm::IntersectionOverObjContained<Dtype>(0.7f));

  MinibatchDataParameter param = this->layer_param_.minibatch_data_param();
  num_batch_ = param.num_batch();
  max_num_patch_ = param.max_num_patch();
  num_gt_ = param.num_gt();
  width_ = param.width();
  if (width_ <= 0) width_ = bottom[0]->width();
  height_ = param.height();
  if (height_ <= 0) height_ = bottom[0]->height();

  std::vector<int> data_shape(4);
  data_shape[0] = num_batch_;
  data_shape[1] = transformed_data_.channels();
  data_shape[2] = height_;
  data_shape[3] = width_;
  top[0]->Reshape(data_shape);

  if (top.size() > 1) { // label
    std::vector<int> label_shape(3);
    label_shape[0] = num_batch_;
    label_shape[1] = 1;
    label_shape[2] = num_gt_;
    top[1]->Reshape(label_shape);
  }

  if (top.size() > 2) { // bbox
    std::vector<int> bbox_shape(3);
    bbox_shape[0] = num_batch_;
    bbox_shape[1] = 4;
    bbox_shape[2] = num_gt_;
    top[2]->Reshape(bbox_shape);
  }
}

template <typename Dtype>
void MinibatchDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  while (minibatch_queue_.Size() <= num_batch_)
    ExtractMinibatchBlob(top.size() > 1, top.size() > 2);

  MakeTopBlob(top);
}

template <typename Dtype>
void MinibatchDataLayer<Dtype>::SelectMinibatch(
    int src_width, int src_height,
    const std::vector<int>& src_label,
    const std::vector<cv::Rect_<Dtype> >& src_bbox,
    std::vector<Minibatch>* minibatch) {
  CHECK_GE(src_width, width_);
  CHECK_GE(src_height, height_);
  CHECK_EQ(src_label.size(), src_bbox.size());
  CHECK(minibatch);

  minibatch->clear();

  int tl_x_range = src_width - width_ + 1;
  int tl_y_range = src_height - height_ + 1;
  int num_minibatch = std::min(max_num_patch_, tl_x_range * tl_y_range);

  // 그냥 n회 추출
  //for (int n = num_minibatch; n--; ) {
  //  int tl_x = Random(0, src_width - width_);
  //  int tl_y = Random(0, src_height - height_);
  //  cv::Rect roi(tl_x, tl_y, width_, height_);
  //  
  //  std::vector<int> new_label;
  //  std::vector<cv::Rect_<Dtype> > new_bbox;
  //  for (int i = 0; i < src_label.size(); ++i) {
  //    if ((*obj_contained_)(roi, src_bbox[i])) {
  //      new_label.push_back(src_label[i]);
  //      new_bbox.push_back(
  //          cv::Rect_<Dtype>(src_bbox[i].x - tl_x,
  //                           src_bbox[i].y - tl_y,
  //                           src_bbox[i].width,
  //                           src_bbox[i].height));
  //    }
  //  }

  //  minibatch->push_back({roi, new_label, new_bbox});
  //}

  // positive scene 이면 positive minibatch만 추출
  //while (minibatch->size() < num_minibatch) {
  //  int tl_x = Random(0, src_width - width_);
  //  int tl_y = Random(0, src_height - height_);
  //  cv::Rect roi(tl_x, tl_y, width_, height_);
  //  
  //  std::vector<int> new_label;
  //  std::vector<cv::Rect_<Dtype> > new_bbox;
  //  for (int i = 0; i < src_label.size(); ++i) {
  //    if ((*obj_contained_)(roi, src_bbox[i])) {
  //      new_label.push_back(src_label[i]);
  //      new_bbox.push_back(
  //          cv::Rect_<Dtype>(src_bbox[i].x - tl_x,
  //                           src_bbox[i].y - tl_y,
  //                           src_bbox[i].width,
  //                           src_bbox[i].height));
  //    }
  //  }

  //  if (!src_label.empty() && new_label.empty())
  //    continue;

  //  minibatch->push_back({roi, new_label, new_bbox});
  //}

  // neg와 pos를 반반 (pos only db시 사용)
  while (minibatch->size() < num_minibatch / 2) {
    int tl_x = Random(0, src_width - width_);
    int tl_y = Random(0, src_height - height_);
    cv::Rect roi(tl_x, tl_y, width_, height_);
    
    std::vector<int> new_label;
    std::vector<cv::Rect_<Dtype> > new_bbox;
    for (int i = 0; i < src_label.size(); ++i) {
      if ((*obj_contained_)(roi, src_bbox[i])) {
        new_label.push_back(src_label[i]);
        new_bbox.push_back(
            cv::Rect_<Dtype>(src_bbox[i].x - tl_x,
                             src_bbox[i].y - tl_y,
                             src_bbox[i].width,
                             src_bbox[i].height));
      }
    }

    //if (!src_label.empty() && new_label.empty())
    //  continue;

    minibatch->push_back({roi, new_label, new_bbox});
  }
  while (minibatch->size() < num_minibatch) {
    int tl_x = Random(0, src_width - width_);
    int tl_y = Random(0, src_height - height_);
    cv::Rect roi(tl_x, tl_y, width_, height_);
    
    std::vector<int> new_label;
    std::vector<cv::Rect_<Dtype> > new_bbox;
    for (int i = 0; i < src_label.size(); ++i) {
      if ((*obj_contained_)(roi, src_bbox[i])) {
        new_label.push_back(src_label[i]);
        new_bbox.push_back(
            cv::Rect_<Dtype>(src_bbox[i].x - tl_x,
                             src_bbox[i].y - tl_y,
                             src_bbox[i].width,
                             src_bbox[i].height));
      }
    }

    if (!src_label.empty() && new_label.empty())
      continue;

    minibatch->push_back({roi, new_label, new_bbox});
  }
}

template <typename Dtype>
void MinibatchDataLayer<Dtype>::MakeDataBlob(
    const Blob<Dtype>& src_data, int batch_idx,
    const cv::Rect& roi, Blob<Dtype>* dst_data) const {
  CHECK_GE(roi.x, 0);
  CHECK_GE(roi.y, 0);
  CHECK_LE(roi.x + roi.width, src_data.width());
  CHECK_LE(roi.y + roi.height, src_data.height());
  CHECK_GE(batch_idx, 0);
  CHECK_LT(batch_idx, src_data.num());
  CHECK(dst_data);
  
  std::vector<int> shape(4);
  shape[0] = 1;
  shape[1] = src_data.channels();
  shape[2] = roi.height;
  shape[3] = roi.width;
  dst_data->Reshape(shape);

  for (int c = 0; c < src_data.channels(); ++c) {
    const Dtype* src_row = src_data.cpu_data() +
        src_data.offset(batch_idx, c, roi.y, roi.x);
    Dtype* dst_row = dst_data->mutable_cpu_data() +
        dst_data->offset(0, c);

    for (int h = roi.height; h--; ) {
      caffe_copy(roi.width, src_row, dst_row);
      src_row += src_data.width();
      dst_row += roi.width;
    }
  }

#ifndef NDEBUG
  int mat_depth = ((sizeof(Dtype) == sizeof(float)) ? CV_32F : CV_64F);
  cv::Mat blue(cv::Size(roi.width, roi.height),
               CV_MAKETYPE(mat_depth, 1),
               dst_data->mutable_cpu_data());
  cv::Mat green(cv::Size(roi.width, roi.height),
                CV_MAKETYPE(mat_depth, 1),
                dst_data->mutable_cpu_data() + dst_data->offset(0, 1));
  cv::Mat red(cv::Size(roi.width, roi.height),
              CV_MAKETYPE(mat_depth, 1),
              dst_data->mutable_cpu_data() + dst_data->offset(0, 2));
  std::vector<cv::Mat> channels(3);
  channels[0] = blue;
  channels[1] = green;
  channels[2] = red;

  cv::Mat merged;
  cv::merge(channels, merged);
#endif // !NDEBUG

}

template <typename Dtype>
void MinibatchDataLayer<Dtype>::MakeLabelBlob(
    const std::vector<int> label, Blob<Dtype>* dst_label) const {
  CHECK_LE(label.size(), num_gt_);
  CHECK(dst_label);

  std::vector<int> shape(3, 1);
  shape[2] = num_gt_;
  dst_label->Reshape(shape);

  Dtype* dst_iter = dst_label->mutable_cpu_data();
  int i;
  for (i = 0; i < label.size(); ++i)
    *dst_iter++ = label[i];
  for (; i < num_gt_; ++i)
    *dst_iter++ = LabelParameter::DUMMY_LABEL;
}

template <typename Dtype>
void MinibatchDataLayer<Dtype>::MakeBBoxBlob(
    const std::vector<cv::Rect_<Dtype> >& bbox,
    Blob<Dtype>* dst_bbox) const {
  CHECK_LE(bbox.size(), num_gt_);
  CHECK(dst_bbox);

  std::vector<int> shape(3);
  shape[0] = 1;
  shape[1] = 4;
  shape[2] = num_gt_;
  dst_bbox->Reshape(shape);

  Dtype* dst_ptr = dst_bbox->mutable_cpu_data();
  Dtype* x_iter = dst_ptr + dst_bbox->offset(0, 0);
  Dtype* y_iter = dst_ptr + dst_bbox->offset(0, 1);
  Dtype* w_iter = dst_ptr + dst_bbox->offset(0, 2);
  Dtype* h_iter = dst_ptr + dst_bbox->offset(0, 3);
  int i;
  for (i = 0; i < bbox.size(); ++i) {
    *x_iter++ = bbox[i].x;
    *y_iter++ = bbox[i].y;
    *w_iter++ = bbox[i].width;
    *h_iter++ = bbox[i].height;
  }
  for (; i < num_gt_; ++i) {
    *x_iter++ = BBoxParameter::DUMMY_VALUE;
    *y_iter++ = BBoxParameter::DUMMY_VALUE;
    *w_iter++ = BBoxParameter::DUMMY_VALUE;
    *h_iter++ = BBoxParameter::DUMMY_VALUE;
  }
}

////template <typename Dtype>
////void MinibatchDataLayer<Dtype>::MakeDataBlob(
////    const Blob<Dtype>& src_data,
////    const std::vector<std::vector<Minibatch> >& minibatch,
////    Blob<Dtype>* dst_data) const {
////  CHECK_EQ(src_data.num(), minibatch.size());
////  CHECK(dst_data);
////
////  int num_minibatch = 0;
////  for (auto iter = minibatch.cbegin(); iter != minibatch.cend(); ++iter)
////    num_minibatch += iter->size();
////
////  std::vector<int> shape(4);
////  shape[0] = num_minibatch;
////  shape[1] = src_data.channels();
////  shape[2] = height_;
////  shape[3] = width_;
////  dst_data->Reshape(shape);
////
////  const Dtype* src_ptr = src_data.cpu_data();
////  Dtype* dst_ptr = dst_data->mutable_cpu_data();
////  int minibatch_idx = 0;
////
////  for (int n = 0; n < src_data.num(); ++n) {
////    for (int m = 0; m < minibatch[n].size(); ++m) {
////      for (int c = 0; c < src_data.channels(); ++c) {
////        const Dtype* src_iter = src_ptr +
////          src_data.offset(n, c, 
////                          minibatch[n][m].roi.y, 
////                          minibatch[n][m].roi.x);
////        Dtype* dst_iter = dst_ptr + dst_data->offset(minibatch_idx, c);
////
////        for (int h = height_; h--;) {
////          caffe_copy(width_, src_iter, dst_iter);
////          src_iter += src_data.width();
////          dst_iter += width_;
////        }
////      }
////
////      ++minibatch_idx;
////    }
////  }
////}
////
////template <typename Dtype>
////void MinibatchDataLayer<Dtype>::MakeLabelBlob(
////    const std::vector<std::vector<Minibatch> >& minibatch,
////    Blob<Dtype>* dst_label) const {
////  CHECK(dst_label);
////
////  int num_label = 0;
////  for (auto i = minibatch.cbegin(); i != minibatch.cend(); ++i)
////    for (auto j = i->cbegin(); j != i->cend(); ++j)
////      num_label += j->label.size();
////
////  std::vector<int> shape(3);
////  shape[0] = num_label;
////  shape[1] = 1;
////  shape[2] = num_gt_;
////  dst_label->Reshape(shape);
////
////  Dtype* label_iter = dst_label->mutable_cpu_data();
////
////  for(auto iter = minibatch)
////  for (int n = dst_label->num(); n--;) {
////    int i;
////    CHECK_LE(minibatch[n].label.size(), num_gt_);
////    for (i = 0; i < minibatch[n].label.size(); ++i)
////      *label_iter++ = minibatch[n].label[i];
////    for (; i < num_gt_; ++i)
////      *label_iter++ = LabelParameter::DUMMY_LABEL;
////  }  
////}
////
////template <typename Dtype>
////void MinibatchDataLayer<Dtype>::MakeBBoxBlob(
////    const std::vector<Minibatch>& minibatch,
////    Blob<Dtype>* dst_bbox) const {
////  CHECK(dst_bbox);
////  CHECK_EQ(dst_bbox->num(), minibatch.size());
////  CHECK_EQ(dst_bbox->channels(), 4);
////  CHECK_EQ(dst_bbox->width(), num_gt_);
////
////  Dtype* bbox_ptr = dst_bbox->mutable_cpu_data();
////  for (int n = dst_bbox->num(); n--; ) {
////    Dtype* x_iter = bbox_ptr + dst_bbox->offset(n, 0);
////    Dtype* y_iter = bbox_ptr + dst_bbox->offset(n, 1);
////    Dtype* w_iter = bbox_ptr + dst_bbox->offset(n, 2);
////    Dtype* h_iter = bbox_ptr + dst_bbox->offset(n, 3);
////
////    int i;
////    CHECK_LE(minibatch[n].bbox.size(), num_gt_);
////    for (i = 0; i < minibatch[n].bbox.size(); ++i) {
////      *x_iter++ = minibatch[n].bbox[i].x;
////      *y_iter++ = minibatch[n].bbox[i].y;
////      *w_iter++ = minibatch[n].bbox[i].width;
////      *h_iter++ = minibatch[n].bbox[i].height;
////    }
////    for (; i < num_gt_; ++i) {
////      *x_iter++ = BBoxParameter::DUMMY_VALUE;
////      *y_iter++ = BBoxParameter::DUMMY_VALUE;
////      *w_iter++ = BBoxParameter::DUMMY_VALUE;
////      *h_iter++ = BBoxParameter::DUMMY_VALUE;
////    }
////  }
////}

template <typename Dtype>
void MinibatchDataLayer<Dtype>::MakeROIBlob(
    const std::vector<std::vector<Minibatch> >& minibatch,
    Blob<Dtype>* dst_roi) const {
  LOG(FATAL) << "Not implemented yet";
}

template <typename Dtype>
void MinibatchDataLayer<Dtype>::ExtractMinibatchBlob(
    bool make_label_blob, bool make_bbox_blob) {
  if (prefetch_current_)
    prefetch_free_.push(prefetch_current_);
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  const Blob<Dtype>& src_data = prefetch_current_->data_;
  std::vector<std::vector<Dtype> > gt_label;
  std::vector<std::vector<cv::Rect_<Dtype> > > gt_bbox;
  BaseImgBBoxDataLayer<Dtype>::ParseLabelBBox(prefetch_current_->label_, 
                                              &gt_label, &gt_bbox);
  CHECK_EQ(src_data.num(), gt_label.size());
  CHECK_EQ(gt_label.size(), gt_bbox.size());
  for (int n = 0; n < src_data.num(); ++n) {
    CHECK_EQ(gt_label[n].size(), gt_bbox[n].size());

    std::vector<int> gt_int_label;
    for (auto iter = gt_label[n].cbegin(); iter != gt_label[n].cend();
         ++iter)
      gt_int_label.push_back(static_cast<int>(*iter));

    std::vector<Minibatch> minibatch;
    SelectMinibatch(src_data.width(), src_data.height(),
                    gt_int_label, gt_bbox[n], &minibatch);

    for (auto m = minibatch.cbegin(); m != minibatch.cend(); ++m) {
      CHECK_EQ(m->label.size(), m->bbox.size());

      MinibatchBlob* minibatch_blob = minibatch_queue_.GetEmptyBin();
      MakeDataBlob(src_data, n, m->roi, &(minibatch_blob->data));
      if (make_label_blob)
        MakeLabelBlob(m->label, &(minibatch_blob->label));
      if (make_bbox_blob)
        MakeBBoxBlob(m->bbox, &(minibatch_blob->bbox));

      minibatch_queue_.Push(minibatch_blob);
    }
  }
}

template <typename Dtype>
void MinibatchDataLayer<Dtype>::MakeTopBlob(
    const vector<Blob<Dtype>*>& top) {
  bool make_label_blob = top.size() > 1;
  bool make_bbox_blob = top.size() > 2;

  CHECK_GE(minibatch_queue_.Size(), num_batch_);

  // 여기 한번 체크할 것.
  // gt 변환 레이어에서 확인한 바, 모든 gt가 -1로 나옴
  // 제대로 뽑히는지, 제대로 forward되는지 체크할 것

  std::vector<int> shape(4);
  shape[0] = num_batch_;
  //shape[1] = (minibatch_queue_.Top())->get()->data.channels();
  shape[2] = height_;
  shape[3] = width_;

  for (int n = 0; n < num_batch_; ++n) {
    auto mb_blob = minibatch_queue_.Top();
    
    Dtype* dst_data = top[0]->mutable_cpu_data() + top[0]->offset(n);
    caffe_copy((*mb_blob)->data.count(), (*mb_blob)->data.cpu_data(),
               dst_data);

    if (make_label_blob) {
      Dtype* dst_label = top[1]->mutable_cpu_data() + top[1]->offset(n);
      caffe_copy((*mb_blob)->label.count(), (*mb_blob)->label.cpu_data(),
                 dst_label);
    }
    
    if (make_bbox_blob) {
      Dtype* dst_bbox = top[2]->mutable_cpu_data() + top[2]->offset(n);
      caffe_copy((*mb_blob)->bbox.count(), (*mb_blob)->bbox.cpu_data(),
                 dst_bbox);
    }

    minibatch_queue_.Pop();
  }
}

#ifdef CPU_ONLY
STUB_GPU(MinibatchDataLayer);
#endif

INSTANTIATE_CLASS(MinibatchDataLayer);
REGISTER_LAYER_CLASS(MinibatchData);
} // namespace caffe