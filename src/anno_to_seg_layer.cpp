#include "anno_to_seg_layer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace caffe
{
template <typename Dtype>
void AnnoToSegLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  AnnoToSegParameter param = this->layer_param_.anno_to_seg_param();

  objectness_ = param.objectness();

  if (objectness_)
    num_label_ = 1;
  else {
    CHECK(param.has_num_label());
    num_label_ = param.num_label();
  }

  bbox_normalized_ = param.bbox_normalized();
  
  in_size_.width = param.in_width();
  in_size_.height = param.in_height();
  
  out_size_.width = param.has_out_width() ? param.out_width() : in_size_.width;
  out_size_.height = param.has_out_height() ? param.out_height() : in_size_.height;
}

template <typename Dtype>
void AnnoToSegLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[1]->channels(), 4);
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());

  std::vector<int> shape(4);
  shape[0] = bottom[0]->num();
  shape[1] = num_label_;
  shape[2] = out_size_.height;
  shape[3] = out_size_.width;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void AnnoToSegLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& label_blob = *(bottom[0]);
  const Blob<Dtype>& bbox_blob = *(bottom[1]);
  Blob<Dtype>& seg_map = *(top[0]);

  const int BATCH_SIZE = label_blob.num();

  std::vector<std::vector<int> > labels;
  std::vector<std::vector<cv::Rect_<Dtype> > > bboxes;
  ParseGT(label_blob, bbox_blob, &labels, &bboxes);

  Dtype* seg_map_ptr = seg_map.mutable_cpu_data();
  caffe_set(seg_map.count(), static_cast<Dtype>(0), seg_map_ptr);

  for (int n = 0; n < BATCH_SIZE; ++n) {
    std::vector<cv::Mat> seg_map_channels(
        num_label_,
        cv::Mat(out_size_, 
                CV_MAKETYPE(sizeof(Dtype) == sizeof(float) ? CV_32F : CV_64F, 1),
                static_cast<Dtype>(0)));

    std::vector<int>& label = labels[n];
    std::vector<cv::Rect_<Dtype> >& bbox = bboxes[n];
    CHECK_EQ(label.size(), bbox.size());

    for (int i = 0; i < label.size(); ++i) {
      //seg_map_channels[label[i] - 1](bbox[i]) = static_cast<Dtype>(1);
      cv::rectangle(seg_map_channels[label[i] - 1], bbox[i],
                    cv::Scalar(1), CV_FILLED);
    }

    for (int c = 0; c < num_label_; ++c) {
#ifndef NDEBUG
      cv::Mat debug = seg_map_channels[c];
#endif
      caffe_copy<Dtype>(out_size_.area(),
                        reinterpret_cast<const Dtype*>(seg_map_channels[c].data),
                        seg_map_ptr + seg_map.offset(n, c));
    }
  }
  
}

template <typename Dtype>
void AnnoToSegLayer<Dtype>::ParseGT(
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
      //if (!discard && !ignore_label_.empty()) {
      //  auto ignore_iter = std::find(ignore_label_.cbegin(),
      //                               ignore_label_.cend(),
      //                               *label_iter);
      //  discard |= (ignore_iter != ignore_label_.cend());
      //}

      if (!discard) {
        int new_label = objectness_ ? 1 : *label_iter;
        CHECK_GT(new_label, 0);
        CHECK_LE(new_label, num_label_);

        cv::Rect_<Dtype> new_bbox(*x_iter, *y_iter, *w_iter, *h_iter);
        if (bbox_normalized_) {
          new_bbox.width = std::exp(new_bbox.width) * in_size_.width;
          new_bbox.height = std::exp(new_bbox.height) * in_size_.height;

          new_bbox.x = (new_bbox.x * in_size_.width) - (new_bbox.width / 2.);
          new_bbox.y = (new_bbox.y * in_size_.height) - (new_bbox.height / 2.);
        }

        Dtype w_scale = out_size_.width / static_cast<Dtype>(in_size_.width);
        Dtype h_scale = out_size_.height / static_cast<Dtype>(in_size_.height);
        new_bbox.x *= w_scale;
        new_bbox.y *= h_scale;
        new_bbox.width *= w_scale;
        new_bbox.height *= h_scale;

        new_bbox.x = std::ceil(new_bbox.x);
        new_bbox.y = std::ceil(new_bbox.y);
        new_bbox.width = std::floor(new_bbox.width);
        new_bbox.height = std::floor(new_bbox.height);

        //new_bbox.x = std::max(static_cast<Dtype>(0), new_bbox.x);
        //new_bbox.x = std::min(new_bbox.x, static_cast<Dtype>(out_size_.width));
        //new_bbox.y = std::max(static_cast<Dtype>(0), new_bbox.y);
        //new_bbox.y = std::min(new_bbox.y, static_cast<Dtype>(out_size_.height));
        //new_bbox.width = std::min(new_bbox.width, out_size_.width - new_bbox.x);
        //new_bbox.width = std::max(static_cast<Dtype>(0), new_bbox.width);
        //new_bbox.height = std::min(new_bbox.height, out_size_.height - new_bbox.y);
        //new_bbox.height = std::max(static_cast<Dtype>(0), new_bbox.height);

        (*label)[n].push_back(new_label);
        (*bbox)[n].push_back(new_bbox);
      }

      ++label_iter;
      ++x_iter;
      ++y_iter;
      ++w_iter;
      ++h_iter;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AnnoToSegLayer);
#endif

INSTANTIATE_CLASS(AnnoToSegLayer);
REGISTER_LAYER_CLASS(AnnoToSeg);

} // namespace caffe
