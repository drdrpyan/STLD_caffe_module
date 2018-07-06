#ifndef TLR_EARLY_REJECTION_GT_LAYER_HPP_
#define TLR_EARLY_REJECTION_GT_LAYER_HPP_

#include "caffe/layer.hpp"

//#include "anno_decoder.hpp"

#include <memory>

namespace caffe
{

template<typename Dtype>
class EarlyRejectionGTLayer : public Layer<Dtype>
{
 public:
  explicit EarlyRejectionGTLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) override;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int ExactNumBottomBlobs() const override;
  virtual int ExactNumTopBlobs() const override;

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
  //std::unique_ptr<bgm::AnnoDecoder<Dtype> > anno_decoder_;
  void HasAnno(const Blob<Dtype>& label_blob,
               std::vector<bool>* has_anno) const;
};

// inline functions
template<typename Dtype>
inline EarlyRejectionGTLayer<Dtype>::EarlyRejectionGTLayer(
    const LayerParameter& param) : Layer<Dtype>(param){

}

template<typename Dtype>
inline const char* EarlyRejectionGTLayer<Dtype>::type() const {
  return "EarlyRejectionGT";
}

template<typename Dtype>
inline int EarlyRejectionGTLayer<Dtype>::ExactNumBottomBlobs() const {
  return 1;
}

template<typename Dtype>
inline int EarlyRejectionGTLayer<Dtype>::ExactNumTopBlobs() const {
  return 1;
}

template<typename Dtype>
inline void EarlyRejectionGTLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template<typename Dtype>
inline void EarlyRejectionGTLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

template<typename Dtype>
inline void EarlyRejectionGTLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}
} // namespace caffe

#endif // !TLR_EARLY_REJECTION_GT_LAYER_HPP_
