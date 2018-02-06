#include "size_filter_layer.hpp"

namespace caffe
{
template <typename Dtype>
void SizeFilterLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const SizeFilterParameter& param = layer_param_.size_filter_param();

  filter_axis_ = param.axis();
  CHECK_GE(filter_axis_, 0);
  CHECK_LE(filter_axis_, 3);

  if (param.has_min()) {
    //CHECK_LE(static_cast<double>(std::numeric_limits<Dtype>::min()),
    //         param.min());
    //CHECK_GE(static_cast<double>(std::numeric_limits<Dtype>::max()),
    //         param.min());
    min_ = param.min();
  }
  else
    min_ = std::numeric_limits<Dtype>::min();

  if (param.has_max()) {
    //CHECK_LE(static_cast<double>(std::numeric_limits<Dtype>::min()),
    //         param.max());
    //CHECK_GE(static_cast<double>(std::numeric_limits<Dtype>::max()),
    //         param.max());
    max_ = param.max();
  }
  else
    max_ = std::numeric_limits<Dtype>::max();
}

template <typename Dtype>
void SizeFilterLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& predict_bbox = *(bottom[0]);
  const Blob<Dtype>& gt_label = *(bottom[1]);
  const Blob<Dtype>& gt_bbox = *(bottom[2]);

  CHECK_EQ(gt_label.channels(), 1);
  CHECK_EQ(gt_bbox.channels(), 4);
  CHECK(predict_bbox.shape() == gt_bbox.shape());

  top[0]->ReshapeLike(*(bottom[0]));
  top[1]->ReshapeLike(*(bottom[1]));
  top[2]->ReshapeLike(*(bottom[2]));
}

template <typename Dtype>
void SizeFilterLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& bbox_in = *(bottom[0]);
  const Blob<Dtype>& gt_label = *(bottom[1]);
  const Blob<Dtype>& gt_bbox = *(bottom[2]);

  Blob<Dtype>& bbox_out = *(top[0]);
  Blob<Dtype>& gt_label_filtered = *(top[1]);
  Blob<Dtype>& gt_bbox_filtered = *(top[2]);

  bbox_out.CopyFrom(bbox_in);
  gt_label_filtered.CopyFrom(gt_label);
  gt_bbox_filtered.CopyFrom(gt_bbox);

  const Dtype* gt_label_ptr = gt_label.cpu_data();
  const Dtype* gt_bbox_ptr = gt_bbox.cpu_data();
  Dtype* bbox_out_ptr = bbox_out.mutable_cpu_data();
  Dtype* gt_label_filtered_ptr = gt_label_filtered.mutable_cpu_data();
  Dtype* gt_bbox_filtered_ptr = gt_bbox_filtered.mutable_cpu_data();

  const int NUM_BATCH = gt_bbox.num();
  const int NUM_ELEM = gt_bbox.width() * gt_bbox.height();
  for (int n = 0; n < NUM_BATCH; ++n) {
    const Dtype* gt_label_iter = gt_label_ptr + gt_label.offset(n);
    const Dtype* gt_bbox_iter = gt_bbox_ptr + gt_bbox.offset(n, filter_axis_);
    Dtype* x_iter = bbox_out_ptr + bbox_out.offset(n, 0);
    Dtype* y_iter = bbox_out_ptr + bbox_out.offset(n, 1);
    Dtype* w_iter = bbox_out_ptr + bbox_out.offset(n, 2);
    Dtype* h_iter = bbox_out_ptr + bbox_out.offset(n, 3);
    Dtype* gt_label_filtered_iter = gt_label_filtered_ptr + gt_label_filtered.offset(n);
    Dtype* gt_x_filtered_iter = gt_bbox_filtered_ptr + gt_bbox_filtered.offset(n, 0);
    Dtype* gt_y_filtered_iter = gt_bbox_filtered_ptr + gt_bbox_filtered.offset(n, 1);
    Dtype* gt_w_filtered_iter = gt_bbox_filtered_ptr + gt_bbox_filtered.offset(n, 2);
    Dtype* gt_h_filtered_iter = gt_bbox_filtered_ptr + gt_bbox_filtered.offset(n, 3);


    for (int i = NUM_ELEM; i--; ) {
      bool reject = false;
      if (*gt_label_iter == LabelParameter::NONE)
        reject = true;
      else if (*gt_bbox_iter < min_ || *gt_bbox_iter >= max_)
        reject = true;

      if(reject) {
      //if (*gt_label_iter == LabelParameter::NONE ||
      //    *gt_bbox_iter < min_ ||
      //    *gt_bbox_iter >= max_) {
        *x_iter = BBoxParameter::DUMMY_VALUE;
        *y_iter = BBoxParameter::DUMMY_VALUE;
        *w_iter = BBoxParameter::DUMMY_VALUE;
        *h_iter = BBoxParameter::DUMMY_VALUE;

        *gt_label_filtered_iter = LabelParameter::NONE;
        *gt_x_filtered_iter = BBoxParameter::DUMMY_VALUE;
        *gt_y_filtered_iter = BBoxParameter::DUMMY_VALUE;
        *gt_w_filtered_iter = BBoxParameter::DUMMY_VALUE;
        *gt_h_filtered_iter = BBoxParameter::DUMMY_VALUE;
      }

      ++gt_label_iter;
      ++gt_bbox_iter;

      ++x_iter;
      ++y_iter;
      ++w_iter;
      ++h_iter;

      ++gt_label_filtered_iter;
      ++gt_x_filtered_iter;
      ++gt_y_filtered_iter;
      ++gt_w_filtered_iter;
      ++gt_h_filtered_iter;
    }
  }

}


#ifdef CPU_ONLY
STUB_GPU(SizeFilterLayer);
#endif

INSTANTIATE_CLASS(SizeFilterLayer);
REGISTER_LAYER_CLASS(SizeFilter);
} // namespace caffe