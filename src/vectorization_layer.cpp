#include "vectorization_layer.hpp"

namespace caffe
{

template <typename Dtype>
VectorizationLayer<Dtype>::VectorizationLayer(
    const LayerParameter& param) 
  : Layer<Dtype>(param),
    TILE_HEIGHT_(param.channel_tiling_param().tile_height()),
    TILE_WIDTH_(param.channel_tiling_param().tile_width()),
    VERTICAL_STRIDE_(param.channel_tiling_param().vertical_stride()),
    HORIZONTAL_STRIDE_(param.channel_tiling_param().horizontal_stride()) {
  CHECK_GT(TILE_HEIGHT_, 0);
  CHECK_GT(TILE_WIDTH_, 0);
  CHECK_GT(VERTICAL_STRIDE_, 0);
  CHECK_GT(HORIZONTAL_STRIDE_, 0);
}

template <typename Dtype>
void VectorizationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); i++) {
    std::vector<int> top_shape;
    ComputeTopShape(*(bottom[i]), &top_shape);
    top[i]->Reshape(top_shape);
  }
}

template <typename Dtype>
VectorizationLayer<Dtype>::ComputeTopShape(
    const std::vector<int>& bottom_shape,
    std::vector<int>* top_shape) const {
  CHECK(top_shape);

  int num_tile_rows = 
      (bottom_shape[2] - TILE_HEIGHT_) / VERTICAL_STRIDE_ + 1;
  int num_tile_cols = 
      (bottom_shape[2] - TILE_WIDTH_) / HORIZONTAL_STRIDE_ + 1;

  top_shape->resize(4);
  (*top_shape)[0] = bottom_shape[0] * num_tile_rows * num_tile_cols;
  (*top_shape)[1] = bottom_shape[1];
  (*top_shape)[2] = TILE_HEIGHT_;
  (*top_shape)[2] = TILE_WIDTH_;
}

} // namespace caffe