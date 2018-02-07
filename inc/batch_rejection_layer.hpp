//#ifndef TLR_BATCH_REJECTION_LAYER_HPP_
//#define TLR_BATCH_REJECTION_LAYER_HPP_
//
//#include "caffe/layer.hpp"
//
//namespace caffe
//{
//
//template <typename Dtype>
//class BatchRejectionLayer : Layer<Dtype>
//{
// public:
//  explicit BatchRejectionLayer(const LayerParameter& param);
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//                          const vector<Blob<Dtype>*>& top) override;
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//                       const vector<Blob<Dtype>*>& top) override;
//  virtual const char* type() const override;
//  virtual int MinBottomBlobs() const override;
//  virtual int MinTopBlobs() const override;
// 
// private:
//  void GetPassIdx(std::vector<int>* pass_idx) const;
//
//  bool rejection_by_threshold_;
//  float threshold_;
//}; // class BatchRejectionLayer
//
//// inline functions
//template <typename Dtype>
//inline BatchRejectionLayer<Dtype>::BatchRejectionLayer(
//    const LayerParameter& param) : Layer<Dtype>(param) {
//
//}
//
//template <typename Dtype>
//inline const char* BatchRejectionLayer<Dtype>::type() const {
//  return "BatchRejection";
//}
//
//template <typename Dtype>
//inline int BatchRejectionLayer<Dtype>::MinBottomBlobs() const {
//  return 2;
//}
//
//template <typename Dtype>
//inline int BatchRejectionLayer<Dtype>::MinTopBlobs() const {
//  return 1;
//}
//
//} // namespace caffe
//#endif // TLR_BATCH_REJECTION_LAYER_HPP_