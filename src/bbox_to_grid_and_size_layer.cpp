#include "bbox_to_grid_and_size_layer.hpp"

namespace caffe
{

template <typename Dtype>
BBoxToGridAndSizeLayer<Dtype>::BBoxToGridAndSizeLayer(
  const LayerParameter& param)
  : Layer<Dtype>(param) {

}

template <typename Dtype>
void BBoxToGridAndSizeLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  bbox_type_ = layer_param().bbox_param().bbox_type();
  CHECK(bbox_type_ == BBoxParameter::VERTICAL ||
        bbox_type_ == BBoxParameter::HORIZONTAL) << "Illegal bbox type.";

  const BBoxToGridAndSizeParameter& param = layer_param().bbox_to_grid_and_size_param();
  CHECK_EQ(param.x_grid().size(), param.y_grid().size());
  x_grid_.clear();
  y_grid_.clear();
  for (int i = 0; i < param.x_grid().size(); i++) {
    x_grid_.push_back(param.x_grid().Get(i));
    y_grid_.push_back(param.y_grid().Get(i));
  }
  std::sort(x_grid_.begin(), x_grid_.end());
  std::sort(y_grid_.begin(), y_grid_.end());

  size_grid_.clear();
  for (int i = 0; i < param.size_grid().size(); i++)
    size_grid_.push_back(param.size_grid().Get(i));
  std::sort(size_grid_.begin(), size_grid_.end());
}

template <typename Dtype>
void BBoxToGridAndSizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& bbox_input = *(bottom[0]);
  Blob<Dtype>& grid_out = *(top[0]);
  Blob<Dtype>& size_out = *(top[1]);

  //std::vector<int> grid_out_shape = bbox_input.shape();
  //grid_out_shape[1] = (x_grid_.size() + 1) * (y_grid_.size() + 1);
  //grid_out.Reshape(grid_out_shape);

  //std::vector<int> size_out_shape = bbox_input.shape();
  //size_out_shape[1] = size_grid_.size() + 1;
  //size_out.Reshape(size_out_shape);

  std::vector<int> out_shape = bbox_input.shape();
  out_shape[1] = 1;
  grid_out.Reshape(out_shape);
  size_out.Reshape(out_shape);
}

template <typename Dtype>
void BBoxToGridAndSizeLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& bbox_input = *(bottom[0]);
  Blob<Dtype>& grid_out = *(top[0]);
  Blob<Dtype>& size_out = *(top[1]);
  
  for (int n = 0; n < bbox_input.num(); n++) {
    const Dtype* x_iter = bbox_input.cpu_data() + bbox_input.offset(n, 0);
    const Dtype* y_iter = bbox_input.cpu_data() + bbox_input.offset(n, 1);
    const Dtype* w_iter = bbox_input.cpu_data() + bbox_input.offset(n, 2);
    const Dtype* h_iter = bbox_input.cpu_data() + bbox_input.offset(n, 3);
    Dtype* grid_out_iter = grid_out.mutable_cpu_data() + grid_out.offset(n);
    Dtype* size_out_iter = size_out.mutable_cpu_data() + size_out.offset(n);

    for (int i = bbox_input.height() * bbox_input.width(); i--; ) {
      *grid_out_iter = GetGridIdx(*x_iter, *y_iter, *w_iter, *h_iter);
      *size_out_iter = GetSize((bbox_type_ == BBoxParameter::HORIZONTAL)
                               ? *w_iter : *h_iter);

      ++x_iter;
      ++y_iter;
      ++w_iter;
      ++h_iter;
      ++grid_out_iter;
      ++size_out_iter;
    }
  }
}

template <typename Dtype>
int BBoxToGridAndSizeLayer<Dtype>::GetGridIdx(Dtype x_min, Dtype y_min,
                                              Dtype width, Dtype height) const {
  Dtype center_x, center_y;
  GetBBoxCenter(x_min, y_min, width, height, &center_x, &center_y);
  
  int x_idx, y_idx;
  for (x_idx = 0; x_idx < x_grid_.size() && center_x >= x_grid_[x_idx]; x_idx++);
  for (y_idx = 0; y_idx < y_grid_.size() && center_y >= y_grid_[y_idx]; y_idx++);

  int grid_idx = y_idx * (x_grid_.size() + 1) + x_idx;

  return grid_idx;
}

template <typename Dtype>
Dtype BBoxToGridAndSizeLayer<Dtype>::GetSize(Dtype size) const {
  if (size_grid_.size() == 0)
    return size;
  else {
    int size_idx;
    for (size_idx = 0; 
         size_idx < size_grid_.size() && size >= size_grid_[size_idx];
         size_idx++);
    return static_cast<Dtype>(size_idx);
  }
}

INSTANTIATE_CLASS(BBoxToGridAndSizeLayer);
REGISTER_LAYER_CLASS(BBoxToGridAndSize);

} // namespace caffe