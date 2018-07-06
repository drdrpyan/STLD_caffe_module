#include "early_rejection_gt_layer.hpp"

namespace caffe
{
template<typename Dtype>
void EarlyRejectionGTLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //anno_decoder_.reset(new bgm::AnnoDecoder<Dtype>);

  //std::vector<int> top_shape(4, 1);
  //top_shape[0] = bottom[0]->num();
}

template<typename Dtype>
void EarlyRejectionGTLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::vector<int> top_shape(4, 1);
  top_shape[0] = bottom[0]->num();
  top[0]->Reshape(top_shape);
}

template<typename Dtype>
void EarlyRejectionGTLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::vector<bool> has_anno;
  HasAnno(*(bottom[0]), &has_anno);

  Dtype* out_ptr = top[0]->mutable_cpu_data();
  for (int i = 0; i < has_anno.size(); ++i)
    out_ptr[i] = has_anno[i] ? 1 : 0;
}

template<typename Dtype>
void EarlyRejectionGTLayer<Dtype>::HasAnno(const Blob<Dtype>& label_blob,
                                           std::vector<bool>* has_anno) const {
  CHECK(has_anno);
  has_anno->resize(label_blob.num());

  const Dtype* label_ptr = label_blob.cpu_data();

  for (int i = 0; i < has_anno->size(); ++i) {
    const Dtype* label_iter = label_ptr + label_blob.offset(i);
    bool no_anno = true;
    for (int j = label_blob.count(1); j-- && no_anno; ) {
      int label = *label_iter++;
      no_anno = (label == LabelParameter::NONE) || (label == LabelParameter::DUMMY_LABEL);
    }

    (*has_anno)[i] = !no_anno;
  }
}

#ifdef CPU_ONLY
STUB_GPU(YOLOV2ResultLayer);
#endif

INSTANTIATE_CLASS(EarlyRejectionGTLayer);
REGISTER_LAYER_CLASS(EarlyRejectionGT);

} // namespace caffe