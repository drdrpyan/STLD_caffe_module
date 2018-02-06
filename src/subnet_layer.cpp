#include "subnet_layer.hpp"

// #define RESHAPE_SUBNET_WHEN_RESHAPE_THIS

namespace caffe
{
template <typename Dtype>
void SubnetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  const SubnetParameter& param = this->layer_param_.subnet_param();

  const std::string& model = param.model();
  Phase phase = param.has_phase() ? param.phase() : this->phase_;
  int level = param.has_level() ? param.level() : 0;
  std::vector<string> stages(param.stage_size());
  for (int i = 0; i < stages.size(); ++i)
    stages[i] = param.stage(i);

  net_.reset(new Net<Dtype>(model, phase, level, &stages));
}

template <typename Dtype>
void SubnetLayer<Dtype>::ShareBlobs(
    const vector<Blob<Dtype>*>& be_shared,
    const vector<Blob<Dtype>*>& to_share) const {
  CHECK_EQ(be_shared.size(), to_share.size());
  for (int i = 0; i < be_shared.size(); ++i) {
    to_share[i]->ReshapeLike(*(be_shared[i]));
    to_share[i]->ShareData(*(be_shared[i]));
    to_share[i]->ShareDiff(*(be_shared[i]));
  }
}

#ifdef CPU_ONLY
STUB_GPU(SubnetLayer);
#endif

INSTANTIATE_CLASS(SubnetLayer);
REGISTER_LAYER_CLASS(Subnet);
} // namespace caffe