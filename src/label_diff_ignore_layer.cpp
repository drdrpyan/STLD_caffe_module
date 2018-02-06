#include "label_diff_ignore_layer.hpp"

namespace caffe
{
template <typename Dtype>
void LabelDiffIgnoreLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  const LabelDiffIgnoreParameter& param = layer_param_.label_diff_ignore_param();

  int num_target = param.ignore_label().size();
  CHECK_EQ(num_target, param.ignore_rate().size());

  ignore_label_.resize(num_target);
  ignore_rate_.resize(num_target);

  for (int i = 0; i < num_target; ++i) {
    ignore_label_[i] = param.ignore_label().Get(i);
    ignore_rate_[i] = param.ignore_rate().Get(i);
    CHECK_GE(ignore_rate_[i], 0.0f);
    CHECK_LE(ignore_rate_[i], 1.0f);
  }

  InitMaskGenerator();

  elem_wise_ = param.elem_wise();
}

//template <typename Dtype>
//void LabelDiffIgnoreLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
//                                          const vector<Blob<Dtype>*>& top) {
//  const Blob<Dtype>& label_probs = *(bottom[0]);
//  const Blob<Dtype>& label_gt = *(bottom[1]);
//  Blob<Dtype>& label_probs_out = *(top[0]);
//
//  CHECK_EQ(label_probs.num(), label_gt.num());
//  CHECK_EQ(label_probs.height(), label_gt.height());
//  CHECK_EQ(label_probs.width(), label_gt.width());
//  CHECK_EQ(label_gt.channels(), 1);
//
//  label_probs_out.ReshapeLike(label_probs);
//}

template <typename Dtype>
void LabelDiffIgnoreLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  if (elem_wise_) {
    CHECK(bottom[0]->shape() == bottom[1]->shape());
    top[0]->ReshapeLike(*(bottom[0]));
  }
  else {
    const Blob<Dtype>& label_probs = *(bottom[0]);
    const Blob<Dtype>& label_gt = *(bottom[1]);
    Blob<Dtype>& label_probs_out = *(top[0]);

    CHECK_EQ(label_probs.num(), label_gt.num());
    CHECK_EQ(label_probs.height(), label_gt.height());
    CHECK_EQ(label_probs.width(), label_gt.width());
    CHECK_EQ(label_gt.channels(), 1);

    label_probs_out.ReshapeLike(label_probs);
  }
}

template <typename Dtype>
void LabelDiffIgnoreLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down,
                                               const vector<Blob<Dtype>*>& bottom) {

  if (elem_wise_) {
    Blob<Dtype>& label_probs = *(bottom[0]);
    const Blob<Dtype>& label_gt = *(bottom[1]);
    const Blob<Dtype>& loss_diff = *(top[0]);

    if (propagate_down[0]) {
      label_probs.CopyFrom(loss_diff, true);

      Dtype* label_probs_iter = label_probs.mutable_cpu_diff();
      const Dtype* gt_iter = label_gt.cpu_data();
      Dtype* diff_iter = label_probs.mutable_cpu_diff();

      for (int i = label_probs.count(); i--; ) {
        if (Reject(*gt_iter))
          *diff_iter = 0;

        ++gt_iter;
        ++diff_iter;
      }
    }
  }
  else {
    Blob<Dtype>& label_probs = *(bottom[0]);
    const Blob<Dtype>& label_gt = *(bottom[1]);
    const Blob<Dtype>& loss_diff = *(top[0]);

    //positive_count = 0;
    //negative_count = 0;
    int rejection_count = 0;

    if (propagate_down[0]) {
      label_probs.CopyFrom(loss_diff, true);

      for (int n = 0; n < label_gt.num(); ++n) {
        const Dtype* gt_iter = label_gt.cpu_data() + label_gt.offset(n);
        for (int h = 0; h < label_gt.height(); ++h) {
          for (int w = 0; w < label_gt.width(); ++w) {
            if (Reject(*gt_iter++)) {
              rejection_count++;
              for (int c = 0; c < label_probs.channels(); ++c) {
                Dtype* diff = label_probs.mutable_cpu_diff() + label_probs.offset(n, c, h, w);
                *diff = 0;
              }
            }
          }
        }
      }
    }
  }
}

//
//int positive_count = 0;
//int negative_count = 0;
//
//template <typename Dtype>
//void LabelDiffIgnoreLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//                                               const vector<bool>& propagate_down,
//                                               const vector<Blob<Dtype>*>& bottom) {
//  Blob<Dtype>& label_probs = *(bottom[0]);
//  const Blob<Dtype>& label_gt = *(bottom[1]);
//  const Blob<Dtype>& loss_diff = *(top[0]);
//
//  positive_count = 0;
//  negative_count = 0;
//  int rejection_count = 0;
//
//  if (propagate_down[0]) {
//    label_probs.CopyFrom(loss_diff, true);
//
//    for (int n = 0; n < label_gt.num(); ++n) {
//      const Dtype* gt_iter = label_gt.cpu_data() + label_gt.offset(n);
//      for (int h = 0; h < label_gt.height(); ++h) {
//        for (int w = 0; w < label_gt.width(); ++w) {
//          if (Reject(*gt_iter++)) {
//            rejection_count++;
//            for (int c = 0; c < label_probs.channels(); ++c) {
//              Dtype* diff = label_probs.mutable_cpu_diff() + label_probs.offset(n, c, h, w);
//              *diff = 0;
//            }
//          }            
//        }
//      }
//    }
//
//    DLOG(INFO) << "Positive count : " << positive_count;
//    DLOG(INFO) << "Negative count : " << negative_count;
//    DLOG(INFO) << "Rejection count : " << rejection_count;
//  }
//}

template <typename Dtype>
void LabelDiffIgnoreLayer<Dtype>::InitMaskGenerator() {
  mask_generator_.clear();
  for (int i = 0; i < ignore_rate_.size(); ++i) {
    boost::bernoulli_distribution<float> random_distribution(ignore_rate_[i]);
    mask_generator_.push_back(RNG(caffe_rng(), random_distribution));
  }
}

template <typename Dtype>
bool LabelDiffIgnoreLayer<Dtype>::Reject(int label) {
  std::vector<int>::const_iterator iter = 
      std::find(ignore_label_.cbegin(), ignore_label_.cend(), label);
  if (iter != ignore_label_.cend()) {
    //negative_count++;
    int idx = std::distance(ignore_label_.cbegin(), iter);
    return mask_generator_[idx](); 
  }
  else {
    //positive_count++;
    return false;
  }
}

#ifdef CPU_ONLY
STUB_GPU(LabelDiffIgnoreLayer);
#endif

INSTANTIATE_CLASS(LabelDiffIgnoreLayer);
REGISTER_LAYER_CLASS(LabelDiffIgnore);

} // namespace caffe