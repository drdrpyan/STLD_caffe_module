#ifndef TLR_HEATMAP_CONCAT_LAYER_HPP_
#define TLR_HEATMAP_CONCAT_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
class HeatmapConcatLayer : public Layer<Dtype>
{
  public:
    explicit HeatmapConcatLayer(const LayerParameter& param);
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual void Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual const char* type() const;

  protected:
    virtual void Forward_cpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual void Forward_gpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;

    /**
    This layer handles input data. It does not backprogate.
    */
    virtual void Backward_cpu(
        const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, 
        const vector<Blob<Dtype>*>& bottom) override;
    /**
    This layer handles input data. It does not backprogate.
    */
    virtual void Backward_gpu(
        const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, 
        const vector<Blob<Dtype>*>& bottom) override;
  
  private:
    void LoadHeatmap(const std::string& heatmap_file);
    //void Concat(input)

    Blob<Dtype> heatmap_;

}; // class HeatmapConcatLayer

// inline functions
template <typename Dtype>
inline HeatmapConcatLayer<Dtype>::HeatmapConcatLayer(
    const LayerParameter& param) 
  : Layer<Dtype>(param) {

}

template <typename Dtype>
inline void HeatmapConcatLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  HeatmapConcatParameter param = 
    this->layer_param_.heatmap_concat_param();
  std::string heatmap_file = param.heatmap_file();
  LoadHeatmap(heatmap_file);
}

template <typename Dtype>
inline const char* HeatmapConcatLayer<Dtype>::type() const {
  return "HeatmapConcat";
}

template <typename Dtype>
inline void HeatmapConcatLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}

template <typename Dtype>
inline void HeatmapConcatLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}
} // namespace caffe
#endif // !TLR_HEATMAP_CONCAT_LAYER_HPP_
