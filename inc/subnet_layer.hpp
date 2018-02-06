#ifndef TLR_SUBNET_LAYER_HPP_
#define TLR_SUBNET_LAYER_HPP_

#include "caffe/layer.hpp"

#include "caffe/net.hpp"

#include <memory>

namespace caffe
{

template <typename Dtype>
class SubnetLayer : public Layer<Dtype>
{
 public:
  explicit SubnetLayer(const LayerParameter& param);
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

  void ShareNetInput(const vector<Blob<Dtype>*>& input) const;
  void ShareNetOutput(const vector<Blob<Dtype>*>& output) const;

  std::unique_ptr<Net<Dtype> > net_;

 private:
  void ShareBlobs(const vector<Blob<Dtype>*>& be_shared,
                  const vector<Blob<Dtype>*>& to_share) const;
}; // class SubnetLayer

// inline functions
template <typename Dtype>
inline SubnetLayer<Dtype>::SubnetLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {
}

template <typename Dtype>
inline void SubnetLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
#ifdef RESHAPE_SUBNET_WHEN_RESHAPE_THIS
  ShareNetInput(bottom);
  net_->Reshape();
  ShareNetOutput(top);
#else
  ShareNetOutput(top);
#endif // RESHAPE_SUBNET_WHEN_RESHAPE_THIS
}

template <typename Dtype>
inline const char* SubnetLayer<Dtype>::SubnetLayer<Dtype>::type() const {
  return "Subnet";
}

template <typename Dtype>
inline void SubnetLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ShareNetInput(bottom);
  net_->Forward();
  ShareNetOutput(top);
}

template <typename Dtype>
inline void SubnetLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void SubnetLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Not implemented yet.";
}

template <typename Dtype>
void SubnetLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Not implemented yet.";
}

template <typename Dtype>
inline void SubnetLayer<Dtype>::ShareNetInput(
    const vector<Blob<Dtype>*>& input) const {
  CHECK_EQ(input.size(), net_->num_inputs());
  ShareBlobs(input, net_->input_blobs());
}

template <typename Dtype>
inline void SubnetLayer<Dtype>::ShareNetOutput(
    const vector<Blob<Dtype>*>& output) const {
  CHECK_EQ(net_->num_outputs(), output.size());
  ShareBlobs(net_->output_blobs(), output);
}


} // namespace caffe

#endif // !TLR_SUBNET_LAYER_HPP_
