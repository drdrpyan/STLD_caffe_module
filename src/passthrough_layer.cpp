//#include "passthrough_layer.hpp"
//
//#include <vector>
//
//namespace caffe
//{
//template <typename Dtype>
//void PassThroughLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
//                                      const vector<Blob<Dtype>*>& top) {
//  CHECK_EQ(bottom[0]->height() % 2, 0);
//  CHECK_EQ(bottom[0]->width() % 2, 0);
//  
//  std::vector<int> top_shape = bottom[0]->shape();
//  top_shape[1] *= 4;
//  top_shape[2] /= 2;
//  top_shape[3] /= 2;
//
//  top[0]->Reshape(top_shape);
//}
//
//} // namespace caffe