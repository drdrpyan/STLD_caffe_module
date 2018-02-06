#include "grid_proposal_loss_layer.hpp"

#include "caffe/util/math_functions.hpp"

namespace caffe
{
template <typename Dtype>
GridProposalLossLayer<Dtype>::GridProposalLossLayer(
    const LayerParameter& param) 
  : LossLayer<Dtype>(param), diff_(),
    loss_layer_(new SigmoidCrossEntropyLossLayer<Dtype>(param)) {
  
}

template <typename Dtype>
void GridProposalLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const GridProposalLossParameter& param 
      = layer_param_.grid_proposal_loss_param();

  noobj_weight_ = param.noobj_weight();

  num_class_ = param.num_class();
  general_obj_proposal_ = param.general_obj_proposal();

  if (general_obj_proposal_) {
    CHECK_EQ(num_class_, 1);

    if (param.obj_weight().size() >= 1) {
      if (param.obj_weight().size() > 1)
        LOG(WARNING) << "Use 'obj_weight[0]' only";
      obj_weight_ = std::vector<float>(1, param.obj_weight().Get(0));
    }
    else
      obj_weight_ = std::vector<float>(1, 1);
  }
  else {
    if (param.obj_weight().size() > 0)
      obj_weight_ = std::vector<float>(num_class_, 1);
    else {
      int num_weight = std::min(num_class_, param.obj_weight().size());
      obj_weight_.resize(num_weight);
      for (int i = 0; i < num_weight; ++i)
        obj_weight_[i] = param.obj_weight().Get(i);
    }
  }

  //loss_bottom_vec_.resize(2);
  //loss_bottom_vec_[0] = bottom[0];
  //transf_gt_.ReshapeLike(*(bottom[0]));
  //loss_bottom_vec_[1] = &transf_gt_;
  ////loss_top_vec_.clear();
  ////loss_top_vec_[0] = loss_output_.get();
  //loss_layer_->SetUp(loss_bottom_vec_, top);
}

template <typename Dtype>
void GridProposalLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& input = *(bottom[0]);
  const Blob<Dtype>& gt = *(bottom[1]);

  CHECK_EQ(input.num_axes(), 3);
  CHECK_EQ(input.num(), gt.num());
  CHECK_EQ(input.channels(), num_class_);
  CHECK_EQ(input.count(2), gt.width() + gt.height());

  diff_.ReshapeLike(input);

  std::vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);

  if (top.size() > 1) 
    top[1]->Reshape(loss_shape);
  if (top.size() > 2)
    top[2]->Reshape(loss_shape);
}

template <typename Dtype>
void GridProposalLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& input = *(bottom[0]);
  const Blob<Dtype>& gt = *(bottom[1]);

  Blob<Dtype> transf_gt;
  TransformGT(gt, &transf_gt);
  TransformGT(gt, &transf_gt_);

  obj_loss_ = std::vector<Dtype>(num_class_, 0);
  obj_count_ = std::vector<int>(num_class_, 0);
  noobj_loss_ = 0;
  noobj_count_ = 0;
  avg_obj_conf_ = 0;
  avg_noobj_conf_ = 0;

  for (int label = 1; label <= num_class_; ++label)
    ComputeDiffLoss(input, transf_gt, label);

  //const Dtype* input_data = input.cpu_data();
  //const Dtype* gt_data = transf_gt.cpu_data();
  //CHECK_EQ(input.num(), transf_gt_.num());
  //CHECK_EQ(input.channels(), transf_gt_.channels());
  //CHECK_EQ(input.height(), transf_gt_.height());
  //CHECK_EQ(input.width(), transf_gt_.width());
  //Dtype* diff_data = diff_.mutable_cpu_data();
  //loss_bottom_vec_[0] = bottom[0];
  //loss_bottom_vec_[1] = &transf_gt_;
  ////loss_top_vec_[0] = top[0];
  //loss_layer_->Forward(loss_bottom_vec_, top);

  

  Dtype loss = 0;
  for (int i = 0; i < obj_loss_.size(); ++i)
    if(obj_count_[i])
      loss += obj_loss_[i] / obj_count_[i];
  if(noobj_count_)
    loss += noobj_loss_ / noobj_count_;

  //int obj_count_sum = caffe_cpu_asum(obj_count_.size(), &(obj_count_[0]));
  int obj_count_sum = 0;
  for (int i = 0; i < obj_count_.size(); ++i)
    obj_count_sum += obj_count_[i];
  if (obj_count_sum)
    avg_obj_conf_ /= obj_count_sum;
  if (noobj_count_)
    avg_noobj_conf_ /= noobj_count_;


  top[0]->mutable_cpu_data()[0] = loss;

  if (top.size() > 1)
    top[1]->mutable_cpu_data()[0] = avg_obj_conf_;
  if (top.size() > 2)
    top[2]->mutable_cpu_data()[0] = avg_noobj_conf_;
}

template <typename Dtype>
void GridProposalLossLayer<Dtype>::Backward_cpu(
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
    caffe_cpu_axpby(bottom[0]->count(), alpha, diff_.cpu_data(),
                    Dtype(0), bottom[0]->mutable_cpu_diff());
  }
  //loss_layer_->Backward(top, propagate_down, loss_bottom_vec_);
}

template <typename Dtype>
void GridProposalLossLayer<Dtype>::ComputeDiffLoss(
    const Blob<Dtype>& input, const Blob<Dtype>& transf_gt,
    Dtype label) {
  const int LABEL_IDX = label - 1;
  const int BATCH_SIZE = input.num();
  const int NUM_ELEM = input.count(1);
  const Dtype OBJ_WEIGHT = obj_weight_[LABEL_IDX];

  const Dtype* input_data = input.cpu_data();
  const Dtype* gt_data = transf_gt.cpu_data();
  Dtype* diff_data = diff_.mutable_cpu_data();

  for (int n = 0; n < BATCH_SIZE; ++n) {
    const Dtype* input_iter = input_data + input.offset(n, label - 1);
    const Dtype* gt_iter = gt_data + transf_gt.offset(n, label - 1);
    Dtype* diff_iter = diff_data + diff_.offset(n, label - 1);

    for (int i = NUM_ELEM; i--;) {
      if (*gt_iter) {
        obj_count_[LABEL_IDX]++;
        obj_loss_[LABEL_IDX] += OBJ_WEIGHT * std::abs(*input_iter - 1);
        *diff_iter = OBJ_WEIGHT * (*input_iter - 1);

        avg_obj_conf_ += *input_iter;
      }
      else {
        noobj_count_++;
        noobj_loss_ += noobj_weight_ * std::abs(*input_iter - 0);
        *diff_iter = noobj_weight_ * (*input_iter - 0);

        avg_noobj_conf_ += *input_iter;
      }

      input_iter++;
      gt_iter++;
      diff_iter++;
    }
  }
}

template <typename Dtype>
void GridProposalLossLayer<Dtype>::TransformGT(
    const Blob<Dtype>& gt, Blob<Dtype>* transformed_gt) const {
  CHECK(transformed_gt);

  transformed_gt->ReshapeLike(diff_);

  const Dtype* gt_data = gt.cpu_data();
  Dtype *transf_data = transformed_gt->mutable_cpu_data();

  if (general_obj_proposal_) {
    for (int n = 0; n < gt.num(); ++n) {
      const Dtype* gt_ptr = gt_data + gt.offset(n);
      Dtype* transf_ptr = transf_data + transformed_gt->offset(n);

      TransformGTRow(gt_ptr, gt.height(), gt.width(),
                     transf_ptr);
      TransformGTCol(gt_ptr, gt.height(), gt.width(),
                     transf_ptr + gt.height());
    }
  }
  else {
    for (int n = 0; n < gt.num(); ++n) {
      const Dtype* gt_ptr = gt_data + gt.offset(n);
      for (int c = 0; c < num_class_; ++c) {
        Dtype* transf_ptr = transf_data + transformed_gt->offset(n, c);

        TransformGTRow(gt_ptr, gt.height(), gt.width(),
                       c + 1, transf_ptr);
        TransformGTCol(gt_ptr, gt.height(), gt.width(),
                       c + 1, transf_ptr + gt.height());
      }
    }
  }
}

template <typename Dtype>
void GridProposalLossLayer<Dtype>::TransformGTRow(
    const Dtype* gt_data, int h, int w,
    Dtype* transformed) const {
  CHECK(transformed);

  const Dtype* gt_data_iter = gt_data;
  for (int i = 0; i < h; ++i) {
    Dtype row_sum = caffe_cpu_asum(w, gt_data_iter);
    transformed[i] = row_sum ? 1 : 0;
    gt_data_iter += w;
  }
}

template <typename Dtype>
void GridProposalLossLayer<Dtype>::TransformGTRow(
    const Dtype* gt_data, int h, int w, Dtype label,
    Dtype* transformed) const {
  CHECK(transformed);

  const Dtype* gt_data_iter = gt_data;
  for (int i = 0; i < h; ++h) {
    int j = 0;
    for (; j < w && (gt_data_iter[j] != label); ++j);
    transformed[i] = (j < w) ? 1 : 0;
    gt_data_iter += w;
  }
}

template <typename Dtype>
void GridProposalLossLayer<Dtype>::TransformGTCol(
    const Dtype* gt_data, int h, int w,
    Dtype* transformed) const {
  CHECK(transformed);

  const Dtype* gt_data_iter = gt_data;
  std::vector<Dtype> sum_row(w, 0);
  for (int i = 0; i < h; ++i) {
    caffe_add(w, gt_data_iter, &(sum_row[0]), &(sum_row[0]));
    gt_data_iter += w;
  }

  for (int i = 0; i < w; ++i)
    transformed[i] = sum_row[i] ? 1 : 0;
}

template <typename Dtype>
void GridProposalLossLayer<Dtype>::TransformGTCol(
    const Dtype* gt_data, int h, int w, Dtype label,
    Dtype* transformed) const {
  CHECK(transformed);

  for (int i = 0; i < w; ++i) {
    const Dtype* gt_data_iter = gt_data + i;

    int j = 0;
    for (; j < h && (*gt_data_iter != label); ++j)
      gt_data_iter += w;

    transformed[i] = (j < h) ? 1 : 0;
  }
}

#ifdef CPU_ONLY
STUB_GPU(GridProposalLossLayer);
#endif

INSTANTIATE_CLASS(GridProposalLossLayer);
REGISTER_LAYER_CLASS(GridProposalLoss);
} // namespace caffe