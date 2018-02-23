//#ifndef TLR_FEATUREMAP_PREFETCHING_DATA_LAYER_HPP_
//#define TLR_FEATUREMAP_PREFETCHING_DATA_LAYER_HPP_
//
//#include "caffe/layer.hpp"
//#include "caffe/internal_thread.hpp"
//
////#include "caffe/util/blocking_queue.hpp"
//#include "caffe/util/db.hpp"
//
////#include "blocking_queue.hpp"
//
//#include <memory>
//
//namespace caffe
//{
//
//template <typename Dtype>
//class FeaturemapPrefetchingDataLayer : public Layer<Dtype>, public InternalThread
//{
//  typedef std::vector<Blob<Dtype> > BlobVec;
//
// public:
//  explicit FeaturemapDataLayer(const LayerParameter& param);
//  virtual ~FeaturemapDataLayer() override;
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//                              const vector<Blob<Dtype>*>& top) override;
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//                       const vector<Blob<Dtype>*>& top) override;
//  virtual const char* type() const override;
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
//  
//  // InternalThread
//  virtual void InternalThreadEntry() override;
//
// private:
//  void LoadBlobVec(BlobVec* blob_vec);
//  void Next();
//  bool Skip();
//
//  std::vector<std::shared_ptr<BlobVec> > prefetch_;
//  bgm::BlockingQueue<BlobVec*> prefetch_free_;
//  bgm::BlockingQueue<BlobVec*> prefetch_full_;
//  BlobVec* prefetch_current_;
//
//  std::shared_ptr<db::DB> db_;
//  std::shared_ptr<db::Cursor> cursor_;
//  uint64_t offset_;
//};
//
//// inline functions
//template <typename Dtype>
//inline FeaturemapDataLayer<Dtype>::~FeaturemapDataLayer() {
//  this->StopInternalThread();
//}
//
//template <typename Dtype>
//inline void FeaturemapDataLayer<Dtype>::Reshape(
//    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//
//}
//
//template <typename Dtype>
//inline const char* FeaturemapDataLayer<Dtype>::type() const {
//  return "FeaturemapData";
//}
//
//template <typename Dtype>
//inline void FeaturemapDataLayer<Dtype>::Forward_gpu(
//    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//  Forward_cpu(bottom, top);
//}
//
//template <typename Dtype>
//inline void FeaturemapDataLayer<Dtype>::Backward_cpu(
//    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
//    const vector<Blob<Dtype>*>& bottom) {
//
//}
//
//template <typename Dtype>
//inline void FeaturemapDataLayer<Dtype>::Backward_gpu(
//    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
//    const vector<Blob<Dtype>*>& bottom) {
//
//}
//
//} // namespace caffe
//
//#endif // !TLR_FEATUREMAP_DATA_LAYER_HPP_
