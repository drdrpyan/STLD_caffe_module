#ifndef FEATUREMAP_SNAPSHOT_LAYER_HPP_
#define FEATUREMAP_SNAPSHOT_LAYER_HPP_

#include "caffe/layer.hpp"

#include "caffe/util/db.hpp"

#include <memory>

namespace caffe
{

template <typename Dtype>
class FeaturemapSnapshotLayer : public Layer<Dtype>
{
  enum {COMMIT_PERIOD = 1000};
 public:
  explicit FeaturemapSnapshotLayer(const LayerParameter& param);
  virtual ~FeaturemapSnapshotLayer() override;
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) override;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) override;
 private:
  std::unique_ptr<caffe::db::DB> db_;
  std::unique_ptr<caffe::db::Transaction> txn_;
  int txn_count_;

}; // class FeaturemapSnapshotLayer

// inline functions
template <typename Dtype>
inline FeaturemapSnapshotLayer<Dtype>::FeaturemapSnapshotLayer(
  const LayerParameter& param) : Layer<Dtype>(param) {
  txn_->Commit();
}

template <typename Dtype>
inline FeaturemapSnapshotLayer<Dtype>::~FeaturemapSnapshotLayer() {

}

template <typename Dtype>
inline void FeaturemapSnapshotLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
inline const char* FeaturemapSnapshotLayer<Dtype>::type() const {
  return "FeaturemapSnapshotLayer";
}

template <typename Dtype>
inline void FeaturemapSnapshotLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void FeaturemapSnapshotLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void FeaturemapSnapshotLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

} // namespace caffe
#endif // !FEATUREMAP_SNAPSHOT_LAYER_HPP_
