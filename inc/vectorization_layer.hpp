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

    void Vectorize_cpu(const Blob<Dtype>& bottom,
                       Blob<Dtype>* top) const;
    //void Vectorize_gpu(const Blob<Dtype>& bottom,
    //                   Blob<Dtype>* top) const;
    void Devectorize_cpu(const Blob<Dtype>& top,
                         Blob <Dtype> *bottom) const;
    //void Devectorize_gpu(const Blob<Dtype>& top,
    //                     Blob <Dtype> *bottom) const;
    void GetDataChIters(const Blob<Dtype>& blob, int n,
                        std::vector<const Dtype*>* ch_iters) const;
    void GetDiffChIters(const Blob<Dtype>& blob, int n,
                        std::vector<Dtype*>* ch_iters) const;
}; // class ReshapingLayer

// inline functions
template <typename Dtype>
inline VectorizationLayer<Dtype>::VectorizationLayer(
    const LayerParameter& param) : Layer<Dtype>(param) {
}

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

template <typename Dtype>
inline void VectorizationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); i++)
    Vectorize_cpu(*(bottom[i]), top[i]);
}

template <typename Dtype>
inline void VectorizationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void VectorizationLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < top.size(); i++)
    if (propagate_down[i])
      Devectorize_cpu(top[i], bottom[i]);
}

template <typename Dtype>
inline void VectorizationLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

} // namespace caffe
#endif // !TLR_VECTORIZATION_LAYER_HPP_
