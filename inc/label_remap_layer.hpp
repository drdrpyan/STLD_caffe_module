#ifndef TLR_LABEL_REMAP_LAYER_HPP_
#define TLR_LABEL_REMAP_LAYER_HPP_

#include "caffe/layer.hpp"

#include <map>

namespace caffe
{

template <typename Dtype>
class LabelRemapLayer : public Layer<Dtype>
{
  public:
    LabelRemapLayer(const LayerParameter& param);
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual void Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual const char* type() const override;
    virtual int MinBottomBlobs() const override;
    virtual bool EqualNumBottomTopBlobs() const override;

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
    void InitRemapTable(const LabelRemapParameter& param);
    int MapLabel(int src) const;
    
    std::map<int, int> label_remap_table_;
}; // class LabelRemapLayer

// inline functions
template <typename Dtype>
inline LabelRemapLayer<Dtype>::LabelRemapLayer(
  const LayerParameter& param) : Layer<Dtype>(param) {

}

template <typename Dtype>
inline void LabelRemapLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for(int i=0; i<bottom.size(); i++)
    top[i]->ReshapeLike(*(bottom[i]));
}

template <typename Dtype>
inline const char* LabelRemapLayer<Dtype>::type() const {
  return "LabelRemap";
}

template <typename Dtype>
inline int LabelRemapLayer<Dtype>::MinBottomBlobs() const {
  return 1;
}

template <typename Dtype>
inline bool LabelRemapLayer<Dtype>::EqualNumBottomTopBlobs() const {
  return true;
}

template <typename Dtype>
inline void LabelRemapLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
inline void LabelRemapLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline void LabelRemapLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template <typename Dtype>
inline int LabelRemapLayer<Dtype>::MapLabel(int src) const {
  //int dst = label_remap_table_.at(src);
  auto dst_iter = label_remap_table_.find(src);
  return dst_iter == label_remap_table_.cend() ? src : dst_iter->second;
}

} // namespace caffe

#endif // !TLR_LABEL_REMPA_LAYER_HPP_
