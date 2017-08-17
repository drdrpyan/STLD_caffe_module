#include "heatmap_concat_layer.hpp"

#include <fstream>
#include <vector>

namespace caffe
{

template <typename Dtype>
void HeatmapConcatLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& input = *(bottom[0]);
  Blob<Dtype>& output = *(top[0]);

  std::vector<int> top_shape(4);
  top_shape[0] = input.num();
  top_shape[1] = input.channels() + heatmap_.channels();
  top_shape[2] = input.height();
  top_shape[3] = input.width();
  
  output.Reshape(top_shape);
}

template <typename Dtype>
void HeatmapConcatLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& input = *(bottom[0]);
  Blob<Dtype>& output = *(top[0]);

  int img_volume = 
    input.channels() * input.height() * input.width();

  for (int n = 0; n < output.num(); n++) {
    Dtype *img_begin = input.mutable_cpu_data() + input.offset(n);
    Dtype *img_end = img_begin + img_volume;
    Dtype *img_dst = output.mutable_cpu_data() + output.offset(n);
    std::copy(img_begin, img_end, img_dst);

    Dtype *heatmap_begin = input.mutable_cpu_data();
    Dtype *heatmap_end = heatmap_begin + heatmap_.count();
    Dtype *heatmap_dst = output.mutable_cpu_data() + 
      output.offset(n, input.channels());
    std::copy(heatmap_begin, heatmap_end, heatmap_dst);
  }
}

// 임시 구현
template <typename Dtype>
void HeatmapConcatLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void HeatmapConcatLayer<Dtype>::LoadHeatmap(
    const std::string& heatmap_file) {
  std::ifstream ifs(heatmap_file, std::ios::binary);
  CHECK(ifs.is_open()) << "There is no file : " << heatmap_file;

  std::vector<int> heatmap_shape(4);
  ifs >> heatmap_shape[0] >> heatmap_shape[1];
  ifs >> heatmap_shape[2] >> heatmap_shape[3];
  heatmap_.Reshape(heatmap_shape);

  ifs.read(heatmap_.mutable_cpu_data(),
           heatmap_.data()->size());
}


} // namespace caffe