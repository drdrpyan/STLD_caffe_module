#include "precision_recall_layer.hpp"  

namespace caffe
{

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  threshold_ = 0.3;
}

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());

  std::vector<int> scalar_shape(0);
  top[0]->Reshape(scalar_shape);
  top[1]->Reshape(scalar_shape);
  top[2]->Reshape(scalar_shape);
  top[3]->Reshape(scalar_shape);
}

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* conf_iter = bottom[0]->cpu_data();
  const Dtype* gt_iter = bottom[1]->cpu_data();

  int tp = 0;
  int fp = 0;
  int fn = 0;

  int pos_cnt = 0;
  int neg_cnt = 0;
  Dtype avg_pos_conf = 0;
  Dtype avg_neg_conf = 0;

  for (int i = bottom[0]->count(); i--; ) {
    Dtype conf = *conf_iter++;
    Dtype gt = *gt_iter++;
    //if (conf > threshold_) {
    //  if (gt != 0)
    //    ++tp;
    //  else
    //    ++fp;
    //}
    //else if (gt != 0)
    //  ++fn;
    if (gt != 0) {
      ++pos_cnt;
      avg_pos_conf += conf;
      if (conf > threshold_)
        ++tp;
      else
        ++fn;
    }
    else {
      ++neg_cnt;
      avg_neg_conf += conf;
      if (conf > threshold_)
        ++fp;
    }
  }

  Dtype precision = tp / static_cast<Dtype>(tp + fp);
  Dtype recall = tp / static_cast<Dtype>(tp + fn);

  if (pos_cnt > 0)
    avg_pos_conf /= pos_cnt;
  if (neg_cnt > 0)
    avg_neg_conf /= neg_cnt;

  (top[0]->mutable_cpu_data())[0] = precision;
  (top[1]->mutable_cpu_data())[0] = recall;
  (top[2]->mutable_cpu_data())[0] = avg_pos_conf;
  (top[3]->mutable_cpu_data())[0] = avg_neg_conf;
}

#ifdef CPU_ONLY
STUB_GPU(PrecisionRecallLayer);
#endif

INSTANTIATE_CLASS(PrecisionRecallLayer);
REGISTER_LAYER_CLASS(PrecisionRecall);

}