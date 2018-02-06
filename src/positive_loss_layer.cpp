#include "positive_loss_layer.hpp"

namespace caffe
{
template <typename Dtype>
void PositiveLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter param(this->layer_param_);
  param.set_type("SigmoidCrossEntropyLoss");
  
  float loss_weight = param.loss_weight(1);
  param.clear_loss_weight();
  param.add_loss_weight(loss_weight);
  loss_layer_ = LayerRegistry<Dtype>::CreateLayer(param);
  loss_bottom_vec_.resize(2);
  loss_bottom_vec_[0] = bottom[0];
  positive_gt_.ReshapeLike(*(bottom[0]));
  loss_bottom_vec_[1] = &positive_gt_;
  loss_top_vec_.resize(1);
  loss_top_vec_[0] = top[0];
  loss_layer_->SetUp(loss_bottom_vec_, loss_top_vec_);

  threshold_ = 0.5;
}

template <typename Dtype>
void PositiveLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->count(), bottom[0]->num());

  std::vector<int> top_shape(0);
  top[0]->Reshape(top_shape);

  positive_gt_.ReshapeLike(*(bottom[0]));

  CHECK_NE(top.size(), 2);
  if (top.size() > 2) {
    top[1]->Reshape(top_shape);
    top[2]->Reshape(top_shape);
  }
}

template <typename Dtype>
void PositiveLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  std::vector<bool> positive_list;
  CheckPositive(*(bottom[1]), &positive_list);
  
  MakePositiveGT(positive_list);

  loss_bottom_vec_[0] = bottom[0];
  loss_bottom_vec_[1] = &positive_gt_;
  loss_top_vec_[0] = top[0];
  loss_layer_->Forward(loss_bottom_vec_, loss_top_vec_);

  if (top.size() > 2) {
    Dtype precision, recall;
    CalcPrRe(*(bottom[0]), positive_list, &precision, &recall);
    top[1]->mutable_cpu_data()[0] = precision;
    top[2]->mutable_cpu_data()[0] = recall;
  }
}

template <typename Dtype>
void PositiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    loss_bottom_vec_[0] = bottom[0];
    loss_top_vec_[0] = top[0];
    std::vector<bool> loss_propagate_down(1);
    loss_propagate_down[0] = true;
    //loss_propagate_down[1] = false;
    loss_layer_->Backward(loss_top_vec_, loss_propagate_down, loss_bottom_vec_);
  }
}

template <typename Dtype>
void PositiveLossLayer<Dtype>::CheckPositive(
    const Blob<Dtype>& label_gt, std::vector<bool>* positive_list) const {
  CHECK(positive_list);

  const int NUM_BATCH = label_gt.num();
  const int NUM_ELEM = label_gt.count() / NUM_BATCH;

  const Dtype* label_gt_ptr = label_gt.cpu_data();

  positive_list->resize(NUM_BATCH);
  for (int i = 0; i < NUM_BATCH; ++i) {
    const Dtype* label_gt_iter = label_gt_ptr + label_gt.offset(i);

    bool empty = true;
    for (int j = 0;j < NUM_ELEM && empty; ++j) {
      empty = (label_gt_iter[j] == LabelParameter::NONE);
      empty = empty || (label_gt_iter[j] == LabelParameter::DUMMY_LABEL);
    }

    (*positive_list)[i] = !empty;
  }
}

template <typename Dtype>
void PositiveLossLayer<Dtype>::MakePositiveGT(
    const std::vector<bool>& positive_list) {
  CHECK_EQ(positive_gt_.count(), positive_list.size());

  Dtype* positive_gt_data = positive_gt_.mutable_cpu_data();
  for (int i = 0; i < positive_list.size(); ++i)
    positive_gt_data[i] = positive_list[i] ? 1 : 0;
}

template <typename Dtype>
void PositiveLossLayer<Dtype>::CalcPrRe(
    const Blob<Dtype>& input, const std::vector<bool>& gt,
    Dtype* precision, Dtype* recall) const {
  CHECK_EQ(input.count(), gt.size());
  CHECK(precision);
  CHECK(recall);

  int tp = 0;
  int fp = 0;
  int fn = 0;

  const Dtype* input_data = input.cpu_data();
  for (int i = 0; i < gt.size(); ++i) {
    Dtype prob = Sigmoid(input_data[i]);
    bool prediction = prob >= threshold_;
    
    if (prediction && gt[i]) tp++;
    else if (prediction && !(gt[i])) fp++;
    else if (!prediction && gt[i]) fn++;
  }

  *precision = tp / static_cast<Dtype>(tp + fp);
  *recall = tp / static_cast<Dtype>(tp + fn);
}

#ifdef CPU_ONLY
STUB_GPU(PositiveLossLayer);
#endif

INSTANTIATE_CLASS(PositiveLossLayer);
REGISTER_LAYER_CLASS(PositiveLoss);

} // namespace caffe
