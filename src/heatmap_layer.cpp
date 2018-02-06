#include "heatmap_layer.hpp"

// debug
#include <opencv2/core.hpp>

namespace caffe
{

template <typename Dtype>
void HeatmapLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  HeatmapParameter param = this->layer_param_.heatmap_param();

  num_label_ = param.num_label();

  bbox_normalized_ = param.bbox_normalized();

  width_ = param.width();
  height_ = param.height();
  CHECK_GT(width_, 0);
  CHECK_GT(height_, 0);

  rows_ = param.rows();
  cols_ = param.cols();
  CHECK_GT(rows_, 0);
  CHECK_GT(cols_, 0);

  x_step_ = (bbox_normalized_ ? 1.0 : width_) / cols_;
  y_step_ = (bbox_normalized_ ? 1.0 : height_) / rows_;
}

template <typename Dtype>
void HeatmapLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& label = *(bottom[0]);
  const Blob<Dtype>& bbox = *(bottom[1]);
  Blob<Dtype>& labelmap = *(top[0]);
  Blob<Dtype>& bboxmap = *(top[1]);

  CHECK_EQ(label.num(), bbox.num());
  CHECK_EQ(label.height(), bbox.height());
  CHECK_EQ(label.width(), bbox.width());

  std::vector<int> top_shape(4);
  top_shape[0] = label.num();
  top_shape[1] = num_label_;
  top_shape[2] = rows_;
  top_shape[3] = cols_;
  labelmap.Reshape(top_shape);

  top_shape[1] = 4;
  bboxmap.Reshape(top_shape);
}

template <typename Dtype>
void HeatmapLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& label = *(bottom[0]);
  const Blob<Dtype>& bbox = *(bottom[1]);
  Blob<Dtype>& labelmap = *(top[0]);
  Blob<Dtype>& bboxmap = *(top[1]);

  caffe_set(labelmap.count(), static_cast<Dtype>(0),
            labelmap.mutable_cpu_data());
  caffe_set(bboxmap.count(), static_cast<Dtype>(-1),
            bboxmap.mutable_cpu_data());

  Dtype* dst_label_ptr = labelmap.mutable_cpu_data();
  Dtype* dst_bbox_ptr = bboxmap.mutable_cpu_data();

  for (int n = 0; n < label.num(); ++n) {
    const Dtype* src_label = label.cpu_data() + label.offset(n);
    const Dtype* src_x = bbox.cpu_data() + bbox.offset(n, 0);
    const Dtype* src_y = bbox.cpu_data() + bbox.offset(n, 1);
    const Dtype* src_w = bbox.cpu_data() + bbox.offset(n, 2);
    const Dtype* src_h = bbox.cpu_data() + bbox.offset(n, 3);

    for (int i = label.height() * label.width(); i--;) {
      if (*src_label != LabelParameter::NONE
          || *src_label != LabelParameter::DUMMY_LABEL) {
        int r, c;
        GetCellPos(*src_x, *src_y, &r, &c);

        Dtype new_x, new_y, new_w, new_h;
        GetNewBBox(*src_x, *src_y, *src_w, *src_h, r, c,
                   &new_x, &new_y, &new_w, &new_h);

        *(dst_label_ptr + labelmap.offset(n, *src_label - 1, r, c)) = 1;
        *(dst_bbox_ptr + bboxmap.offset(n, 0, r, c)) = new_x;
        *(dst_bbox_ptr + bboxmap.offset(n, 1, r, c)) = new_y;
        *(dst_bbox_ptr + bboxmap.offset(n, 2, r, c)) = new_w;
        *(dst_bbox_ptr + bboxmap.offset(n, 3, r, c)) = new_h;
      }

      ++src_label;
      ++src_x;
      ++src_y;
      ++src_w;
      ++src_h;
    }

    //// debug
    //cv::Mat labelmap_mat(cv::Size(rows_, cols_), CV_32FC1, 
    //                     dst_label_ptr + labelmap.offset();
    //cv::Mat labelmap_mat(cv::Size(rows_, cols_), CV_32FC1, dst_label_ptr);
    //cv::Mat labelmap_mat(cv::Size(rows_, cols_), CV_32FC1, dst_label_ptr);
    //cv::Mat labelmap_mat(cv::Size(rows_, cols_), CV_32FC1, dst_label_ptr);
    //cv::Mat labelmap_mat(cv::Size(rows_, cols_), CV_32FC1, dst_label_ptr);
  }
}

template <typename Dtype>
void HeatmapLayer<Dtype>::GetCellPos(Dtype x, Dtype y, 
                                     int* r, int* c) const {
  CHECK(r);
  CHECK(c);

  int i;
  for (i = 0;
       (i < rows_) && (y > (i + 1)*y_step_); 
       ++i);
  *r = i;
  for (i = 0;
       (i < cols_) && (x > (i + 1)*x_step_); 
       ++i);
  *c = i;
}

template <typename Dtype>
void HeatmapLayer<Dtype>::GetNewBBox(
    Dtype src_x, Dtype src_y,
    Dtype src_w, Dtype src_h,
    int pos_r, int pos_c,
    Dtype* dst_x, Dtype* dst_y,
    Dtype* dst_w, Dtype* dst_h) const {
  CHECK(dst_x);
  CHECK(dst_y);
  CHECK(dst_w);
  CHECK(dst_h);

  *dst_x = src_x - pos_c * x_step_;
  *dst_y = src_y - pos_r * y_step_;

  if (bbox_normalized_) {
    *dst_w = std::log(std::exp(src_w) * cols_);
    *dst_h = std::log(std::exp(src_h) * rows_);
  }
  else {
    *dst_w = src_w * cols_; // src_w * width_ / (width_ / cols_);
    *dst_h = src_h * rows_;
  }
}

//template <typename Dtype>
//void HeatmapLayer<Dtype>::GetCenter(
//    Dtype x, Dtype y, Dtype w, Dtype h,
//    Dtype* center_x, Dtype* center_y) const {
//
//}

#ifdef CPU_ONLY
STUB_GPU(HeatmapLayer);
#endif

INSTANTIATE_CLASS(HeatmapLayer);
REGISTER_LAYER_CLASS(Heatmap);

} // namespace caffe