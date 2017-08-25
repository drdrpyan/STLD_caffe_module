#ifndef TLR_CHANNEL_TILING_LAYER_HPP_
#define TLR_CHANNEL_TILING_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class ChannelTilingLayer : public Layer<Dtype>
{
  public:
    explicit ChannelTilingLayer(const LayerParameter& param);
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
    const int TILE_HEIGHT_;
    const int TILE_WIDTH_;
    const int VERTICAL_STRIDE_;
    const int HORIZONTAL_STRIDE_;
}; // class ReshapingLayer

// inline functions
template <typename Dtype>
const char* ChannelTilingLayer<Dtype>::type() const {
  return "ChannelTiling";
}

template <typename Dtype>
bool ChannelTilingLayer<Dtype>::EqualNumBottomTopBlobs() const {
  return true;
}
} // namespace caffe
#endif // !TLR_CHANNEL_TILING_LAYER_HPP_
