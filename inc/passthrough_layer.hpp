//#ifndef TLR_PASSTHROUGH_LAYER_HPP_
//#define TLR_PASSTHROUGH_LAYER_HPP_
//
//#include "caffe/layer.hpp"
//
//namespace caffe
//{
//
//template <typename Dtype>
//class PassThroughLayer : Layer<Dtype>
//{
// public:
//  explicit PassThroughLayer(const LayerParameter& param);
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//                          const vector<Blob<Dtype>*>& top) override;
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//                       const vector<Blob<Dtype>*>& top) override;
//  virtual const char* type() const override;
//  virtual int ExactNumBottomBlobs() const override;
//  virtual int ExactNumTopBlobs() const override;
//
// protected:
//  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top) override;
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top) override;
//  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down, 
//                            const vector<Blob<Dtype>*>& bottom) override;
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down, 
//                            const vector<Blob<Dtype>*>& bottom) override;
//};
//
//// inline functions
//template <typename Dtype>
//inline const char* PassThroughLayer<Dtype>::type() const {
//  return "PassThrough";
//}
//
//
//} // namespace caffe
//#endif // !TLR_PASSTHROUGH_LAYER_HPP_