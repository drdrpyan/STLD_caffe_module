#include "sliding_window_input_layer.hpp"

#include "caffe/util/math_functions.hpp"

namespace caffe
{

template <typename Dtype>
SlidingWindowInputLayer<Dtype>::SlidingWindowInputLayer(
    const LayerParameter& param)
  : Layer<Dtype>(param), 
    WINDOW_WIDTH_(param.sliding_window_param().window_width()),
    WINDOW_HEIGHT_(param.sliding_window_param().window_height()),
    HORIZONTAL_STRIDE_(param.sliding_window_param().horizontal_stride()),
    VERTICAL_STRIDE_(param.sliding_window_param().vertical_stride()),
    WIN_NORMALIZATION_(param.sliding_window_param().window_normalization()),
    num_img_(0), input_width_(0), input_height_(0) {
  CHECK_GT(WINDOW_WIDTH_, 0);
  CHECK_GT(WINDOW_HEIGHT_, 0);
  CHECK_GE(HORIZONTAL_STRIDE_, 0);
  CHECK_GE(VERTICAL_STRIDE_, 0);
}

template <typename Dtype>
void SlidingWindowInputLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& input_img = *(bottom[0]);
  top[0]->ReshapeLike(input_img);

  if (top.size() > 1) {
    UpdateOffsets(input_img.num(), input_img.width(), input_img.height());
    top[1]->ReshapeLike(offsets_);
  }

}

template <typename Dtype>
void SlidingWindowInputLayer<Dtype>::UpdateOffsets(int num_img,
                                                   int input_width,
                                                   int input_height) {
  CHECK_GT(num_img, 0);
  CHECK_GT(input_width, 0);
  CHECK_GT(input_height, 0);

  if (num_img_ != num_img || 
      input_width_ != input_width || 
      input_height_ != input_height) {
    num_img_ = num_img;
    input_width_ = input_width;
    input_height_ = input_height;
    ComputeOffsets();
  }
}

template <typename Dtype>
void SlidingWindowInputLayer<Dtype>::ComputeOffsets() {
  CHECK_GE(input_width_, WINDOW_WIDTH_);
  CHECK_GE(input_height_, WINDOW_HEIGHT_);

  std::vector<int> offset_shape(4);
  offset_shape[0] = num_img_;
  offset_shape[1] = 4;
  //offset_shape[2] = input_height_ / WINDOW_HEIGHT_;
  //offset_shape[3] = input_width_ / WINDOW_WIDTH_;
  offset_shape[2] = ((input_height_ - WINDOW_HEIGHT_) / HORIZONTAL_STRIDE_) + 1;
  offset_shape[3] = ((input_width_ - WINDOW_WIDTH_) / VERTICAL_STRIDE_) + 1;
  offsets_.Reshape(offset_shape);

  const int& WIDTH = offset_shape[2];
  const int& HEIGHT = offset_shape[3];

  // offset x
  Dtype* offset_x_iter = offsets_.mutable_cpu_data();
  Dtype stride_x = 
      WIN_NORMALIZATION_ ? WINDOW_WIDTH_ / static_cast<Dtype>(input_width_) : WINDOW_WIDTH_;
  Dtype offset_x = 0;
  for (int i = 0; i < WIDTH; i++) {
    offset_x_iter[i] = offset_x;
    offset_x += stride_x;
  }
  for (int i = HEIGHT - 1; i--; ) {
    caffe_copy(WIDTH, offset_x_iter, offset_x_iter + WIDTH);
    offset_x_iter += WIDTH;
  }

  // offset y
  Dtype* offset_y_iter = offsets_.mutable_cpu_data() + offsets_.offset(0, 1);
  Dtype stride_y = WIN_NORMALIZATION_ ? WINDOW_HEIGHT_ / static_cast<Dtype>(input_height_) : WINDOW_HEIGHT_;
  Dtype offset_y = 0;
  for (int i = HEIGHT; i--; ) {
    caffe_set(WIDTH, offset_y, offset_y_iter);
    offset_y += stride_y;
    offset_y_iter += WIDTH;
  }

  // window width
  Dtype* win_width_iter = offsets_.mutable_cpu_data() + offsets_.offset(0, 2);
  Dtype win_width = 
      (WIN_NORMALIZATION_) ? WINDOW_WIDTH_ / static_cast<Dtype>(input_width_) : WINDOW_WIDTH_;
  caffe_set(WIDTH * HEIGHT, win_width, win_width_iter);

  // window height
  Dtype* win_height_iter = offsets_.mutable_cpu_data() + offsets_.offset(0, 3);
  Dtype win_height = 
      (WIN_NORMALIZATION_) ? WINDOW_HEIGHT_ / static_cast<Dtype>(input_height_) : WINDOW_HEIGHT_;
  caffe_set(WIDTH * HEIGHT, win_height, win_height_iter);

  // copy along n axis
  Dtype* item_iter = offsets_.mutable_cpu_data();
  int item_size = offset_shape[1] * offset_shape[2] * offset_shape[3];
  for (int i = offsets_.num() - 1; i--;) {
    caffe_copy(item_size, item_iter, item_iter + item_size);
    item_iter += item_size;
  }
}

#ifdef CPU_ONLY
STUB_GPU(SlidingWindowInputLayer);
#endif

INSTANTIATE_CLASS(SlidingWindowInputLayer);
REGISTER_LAYER_CLASS(SlidingWindowInput);

} // namespace caffe