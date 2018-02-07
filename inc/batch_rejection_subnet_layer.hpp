#ifndef TLR_BATCH_REJECTION_SUBNET_LAYER_HPP_
#define TLR_BATCH_REJECTION_SUBNET_LAYER_HPP_

#include "subnet_layer.hpp"

namespace caffe
{

template <typename Dtype>
class BatchRejectionSubnetLayer : public SubnetLayer<Dtype>
{
 public:
  explicit BatchRejectionSubnetLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int MinBottomBlobs() const override;
  virtual int MinTopBlobs() const override;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
 private:
  void Forward(bool copy_gpu,
               const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top);
  void ForwardEmpty(bool copy_gpu,
                    const std::vector<Blob<Dtype>*>& top) const;
  void ForwardSubnet(bool copy_gpu, 
                     const std::vector<int>& rejeciton_idx,
                     const std::vector<Blob<Dtype>*>& bottom,
                     const std::vector<Blob<Dtype>*>& top);
  void ForwardRejectionInfo(const std::vector<int>& rejection_idx,
                            Blob<Dtype>* top) const;
  void GetSubnetBottomTop(const vector<Blob<Dtype>*>& bottom, 
                          const vector<Blob<Dtype>*>& top,
                          vector<Blob<Dtype>*>* subnet_bottom,
                          vector<Blob<Dtype>*>* subnet_top) const;
  void GetRejectionIdx(const Blob<Dtype>& rejection_info,
                       std::vector<int>* rejection_idx) const;
  void RejectBatch(const Blob<Dtype>& src,
                   const std::vector<int>& rejection_idx,
                   bool copy_gpu, Blob<Dtype>* dst) const;
  void InsertDummyBatch(const Blob<Dtype>& src,
                        const std::vector<int> rejection_idx,
                        bool copy_gpu, Blob<Dtype>* dst);

  std::vector<Blob<Dtype>*> subnet_output_blobs_ptr_;
  std::vector<Blob<Dtype> > subnet_output_blobs_;

  bool rejection_by_threshold_;
  float threshold_;
}; // class BatchRejectionSubnetLayer

// inline functions
template <typename Dtype>
inline BatchRejectionSubnetLayer<Dtype>::BatchRejectionSubnetLayer(
    const LayerParameter& param) : SubnetLayer<Dtype>(param) {

}

template <typename Dtype>
inline const char* BatchRejectionSubnetLayer<Dtype>::type() const {
  return "BatchRejecitonSubnet";
}

template <typename Dtype>
inline int BatchRejectionSubnetLayer<Dtype>::MinBottomBlobs() const {
  return 2;
}

template <typename Dtype>
inline int BatchRejectionSubnetLayer<Dtype>::MinTopBlobs() const {
  return 1;
}

} // namespace caffe
#endif // !TLR_BATCH_REJECTION_SUBNET_LAYER_HPP_
