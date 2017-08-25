#ifndef TLR_VECTORIZATION_LAYER_HPP_
#define TLR_VECTORIZATION_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class VectorizationLayer : public Layer<Dtype>
{
  public:
    explicit VectorizationLayer(const LayerParameter& param);
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top) override;
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top) override;
    virtual const char* type() const override;

    virtual bool EqualNumBottomTopBlobs() const override;

  protected:
      virtual void Forward_cpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;

    virtual void Forward_gpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;

    virtual void Backward_cpu(
        const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, 
        const vector<Blob<Dtype>*>& bottom) override;

    virtual void Backward_gpu(
        const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, 
        const vector<Blob<Dtype>*>& bottom) override;

  private:
    void ComputeTopShape(const std::vector<int>& bottom_shape,
                         std::vector<int>* top_shape) const;
}; // class ReshapingLayer

// inline functions
template <typename Dtype>
inline void VectorizationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
inline const char* VectorizationLayer<Dtype>::type() const {
  return "Vectorization";
}

template <typename Dtype>
inline bool VectorizationLayer<Dtype>::EqualNumBottomTopBlobs() const {
  return true;
}
} // namespace caffe
#endif // !TLR_VECTORIZATION_LAYER_HPP_
