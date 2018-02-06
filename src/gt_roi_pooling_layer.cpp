#include "gt_roi_pooling_layer.hpp"

#include "caffe/util/math_functions.hpp"

namespace caffe
{
template <typename Dtype>
void GTROIPoolingLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "x";

  gt_encoder_.reset(new bgm::AnnoEncoder<Dtype>);
  gt_decoder_.reset(new bgm::AnnoDecoder<Dtype>);

  uniform_rng_ = bgm::UniformIntegerRNG<int>::GetInstance();

  obj_contained_.reset(new bgm::IntersectionOverObjContained<Dtype>);
}

template <typename Dtype>
void GTROIPoolingLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<int> top_shape(4);
  top_shape[0] = bottom[0]->num() * (num_pos_ + num_neg_);
  top_shape[1] = bottom[0]->channels();
  top_shape[2] = roi_size_.height;
  top_shape[3] = roi_size_.width;
  top[0]->Reshape(top_shape);

  top[1]->ReshapeLike(*(bottom[1]));
}

template <typename Dtype>
void GTROIPoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<std::vector<int> > roi_label;
  std::vector<std::vector<cv::Rect_<Dtype> > > roi_bbox;
  PickRandomROI(bottom, &roi_label, &roi_bbox);

  PoolROI_cpu(*(bottom[0]), top[0]);
  MakeGTTop(roi_label, roi_bbox, top[1], top[2]);
}

template <typename Dtype>
void GTROIPoolingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0])
    return;

  Dtype* bot_diff_ptr = bottom[0]->mutable_cpu_diff();
  caffe_set<Dtype>(bottom[0]->count(), static_cast<Dtype>(0),
                   bot_diff_ptr);

  const Dtype* top_diff_ptr = top[0]->cpu_diff();

  for (auto iter = roi_relation_.cbegin();
       iter != roi_relation_.cend(); ++iter) {

    for (int c = 0; c < bottom[0]->channels(); ++c) {
      int bot_offset = bottom[0]->offset(iter->bot_idx, c, iter->offset_y, iter->offset_x);
      Dtype* bot_diff_iter = bot_diff_ptr + bot_offset;

      const Dtype* top_diff_iter = top_diff_ptr + top[0]->offset(iter->top_idx, c);

      for (int h = roi_size_.height; h--;) {
        caffe_axpy(roi_size_.width, static_cast<Dtype>(1),
                   top_diff_iter, bot_diff_iter);
        bot_diff_iter += bottom[0]->width();
        top_diff_iter += roi_size_.width;
      }
    }
  }
}

template <typename Dtype>
void GTROIPoolingLayer<Dtype>::PickRandomROI(
    const std::vector<Blob<Dtype>*>& bottom,
    std::vector<std::vector<int> >* roi_label,
    std::vector<std::vector<cv::Rect_<Dtype> > >* roi_bbox) {
  CHECK(roi_label);
  CHECK(roi_bbox);
  int num_roi = (num_pos_ + num_neg_) * bottom[0]->num();
  roi_label->assign(num_roi, std::vector<int>());
  roi_bbox->assign(num_roi, std::vector<cv::Rect_<Dtype> >());

  std::vector<std::vector<int> > gt_label;
  std::vector<std::vector<cv::Rect_<Dtype> > > gt_bbox;
  std::vector<Blob<Dtype>*> gt_bottom(2);
  gt_bottom[0] = bottom[1];
  gt_bottom[1] = bottom[2];

  gt_decoder_->Decode(gt_bottom, &gt_label, &gt_bbox);

  int top_idx = 0;
  if (!pool_each_gt_) {
    for (int n = 0; n < gt_label.size(); ++n) {
      if (gt_bbox[n].size() == 0) {
        for (int i = num_pos_ + num_neg_; i--;) {
          int x = uniform_rng_->Random(0, bottom[0]->width() - roi_size_.width);
          int y = uniform_rng_->Random(0, bottom[0]->height() - roi_size_.height);
          roi_relation_.push_back({n, top_idx++, x, y});
        }
      }
      else {
        std::vector<cv::Rect_<Dtype> > fm_gt;
        ImgBBoxToFMBBox(gt_bbox[n],
                        bottom[0]->width(), bottom[0]->height(),
                        &fm_gt);

        for (int pos = num_pos_; pos--; ) {
          cv::Point roi_offset = GetRandomPosROI(fm_gt);

          std::vector<int> contained_idx;
          GetContainedBBox(cv::Rect(roi_offset, roi_size_),
                           fm_gt, &contained_idx);

          for (auto iter = contained_idx.cbegin();
               iter != contained_idx.cend(); ++iter) {
            (*roi_label)[top_idx].push_back(gt_label[n][*iter]);

            cv::Rect_<Dtype> contained_bbox = gt_bbox[n][*iter];
            contained_bbox.x -= roi_offset.x;
            contained_bbox.y -= roi_offset.y;
            (*roi_bbox)[top_idx].push_back(contained_bbox);
          }

          roi_relation_.push_back({n, top_idx++, roi_offset.x, roi_offset.y});
        }
        for (int neg = num_neg_; neg--; ) {
          int x = uniform_rng_->Random(0, bottom[0]->width() - roi_size_.width);
          int y = uniform_rng_->Random(0, bottom[0]->height() - roi_size_.height);
          
          std::vector<int> contained_idx;
          GetContainedBBox(cv::Rect(cv::Point(x, y), roi_size_),
                           fm_gt, &contained_idx);

          for (auto iter = contained_idx.cbegin();
               iter != contained_idx.cend(); ++iter) {
            (*roi_label)[top_idx].push_back(gt_label[n][*iter]);

            cv::Rect_<Dtype> contained_bbox = gt_bbox[n][*iter];
            contained_bbox.x -= x;
            contained_bbox.y -= y;
            (*roi_bbox)[top_idx].push_back(contained_bbox);
          }

          roi_relation_.push_back({n, top_idx++, x, y});
        }
      }
    }
  }
  else
    LOG(FATAL) << "Not implemented yet.";
}

template <typename Dtype>
void GTROIPoolingLayer<Dtype>::ImgBBoxToFMBBox(
    const std::vector<cv::Rect_<Dtype> >& img_bbox,
    int fm_width, int fm_height,
    std::vector<cv::Rect_<Dtype> >* fm_bbox) const {
  CHECK_GT(fm_width, 0);
  CHECK_GT(fm_height, 0);
  CHECK(fm_bbox);

  fm_bbox->assign(img_bbox.cbegin(), img_bbox.cend());

  Dtype w_scale = fm_width / static_cast<Dtype>(img_size_.width);
  Dtype h_scale = fm_height / static_cast<Dtype>(img_size_.height);

  for (auto iter = fm_bbox->begin(); iter != fm_bbox->end(); ++iter) {
    iter->x *= w_scale;
    iter->y *= h_scale;
    iter->width *= w_scale;
    iter->height *= h_scale;
  }
}

template <typename Dtype>
cv::Point GTROIPoolingLayer<Dtype>::GetRandomPosROI(
    const std::vector<cv::Rect_<Dtype> >& fm_bbox) {
  CHECK_GT(fm_bbox.size(), 0);

  int target_idx = uniform_rng_->Random(0, fm_bbox.size() - 1);
  const cv::Rect_<Dtype>& target = fm_bbox[target_idx];
  int x_min = std::ceil(target.x + target.width) - roi_size_.width;
  int x_max = std::floor(target.x);
  CHECK_LE(x_min, x_max);
  int y_min = std::ceil(target.y + target.height) - roi_size_.height;
  int y_max = std::floor(target.y);
  CHECK_LE(y_min, y_max);

  int x = uniform_rng_->Random(x_min, x_max);
  int y = uniform_rng_->Random(y_min, y_max);
  return cv::Point(x, y);
}

template <typename Dtype>
void GTROIPoolingLayer<Dtype>::GetContainedBBox(
    const cv::Rect& background,
    const std::vector<cv::Rect_<Dtype> >& bbox,
    std::vector<int>* idx) const {
  CHECK(idx);
  idx->clear();

  cv::Rect_<Dtype> bg(background.x, background.y,
                      background.width, background.height);
  for (int i = 0; i < bbox.size(); ++i) {
    if ((*obj_contained_)(bg, bbox[i]))
      idx->push_back(i);
  }
}

template <typename Dtype>
void GTROIPoolingLayer<Dtype>::MakeGTTop(
    const std::vector<std::vector<int> >& roi_label,
    const std::vector<std::vector<cv::Rect_<Dtype> > >& roi_bbox,
    Blob<Dtype>* label_top, Blob<Dtype>* bbox_top) const {
  CHECK(label_top);
  CHECK(bbox_top);

  std::vector<caffe::Blob<Dtype>*> top_blobs(2);
  top_blobs[0] = label_top;
  top_blobs[1] = bbox_top;

  gt_encoder_->Encode(roi_label, roi_bbox, top_blobs);
}

template <typename Dtype>
void GTROIPoolingLayer<Dtype>::PoolROI_cpu(
    const Blob<Dtype>& bottom, Blob<Dtype>* top) const {
  const Dtype* bot_ptr = bottom.cpu_data();
  Dtype* top_ptr = top->mutable_cpu_data();

  for (auto iter = roi_relation_.cbegin(); iter != roi_relation_.cend();
       ++iter) {
    for (int c = 0; c < bottom.channels(); ++c) {
      int bot_offset = bottom.offset(iter->bot_idx, c, iter->offset_y, iter->offset_x);
      const Dtype* bot_iter = bot_ptr + bot_offset;
      Dtype* top_iter = top_ptr + top->offset(iter->top_idx);

      for (int h = roi_size_.height; h--;) {
        caffe_copy(roi_size_.width, bot_iter, top_iter);
        bot_iter += bottom.width();
        top_iter += roi_size_.width;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GTROIPoolingLayer);
#endif

INSTANTIATE_CLASS(GTROIPoolingLayer);
REGISTER_LAYER_CLASS(GTROIPooling);

} // namespace caffe