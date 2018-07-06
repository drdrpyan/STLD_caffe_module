#ifndef TLR_SUBWIN_DATA_2_LAYER_HPP_
#define TLR_SUBWIN_DATA_2_LAYER_HPP_

//#include "base_img_bbox_data_layer.hpp"
#include "subwin_data_layer.hpp"

#include "anno_encoder.hpp"
#include "anno_decoder.hpp"

#include <opencv2/core.hpp>

#include <memory>

namespace caffe
{

template <typename Dtype>
class SubwinData2Layer : public SubwinDataLayer<Dtype>
{
 public:
  explicit SubwinData2Layer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) override;
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) override;
  //virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
  //                     const vector<Blob<Dtype>*>& top) override;
  virtual const char* type() const override;
  virtual int MinTopBlobs() const override;
  virtual int MaxTopBlobs() const override;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) override;
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //                          const vector<bool>& propagate_down, 
  //                          const vector<Blob<Dtype> *> &bottom) override;

 private:
  //void ReshapeTop(const Blob<Dtype>& src_data,
  //                const Blob<Dtype>& src_label,
  //                const std::vector<Blob<Dtype>*>& top) const;
  //void ForwardCroppedImg_cpu(const Blob<Dtype>& src, Blob<Dtype>* dst) const;
  //void ForwardCroppedImg_gpu(const Blob<Dtype>& src, Blob<Dtype>* dst) const;
  //void ForwardLabelBBox(const Blob<Dtype>& src, 
  //                      Blob<Dtype>* label_dst, Blob<Dtype>* bbox_dst) const;
  //void ForwardWholeImg(const Blob<Dtype>& src, Blob<Dtype>* dst) const;

  void Fetch_cpu(const vector<Blob<Dtype>*>& bottom);
  void Fetch_gpu(const vector<Blob<Dtype>*>& bottom);

  void ForwardAnno(const std::vector<std::vector<int> > & label,
                   const std::vector<std::vector<cv::Rect_<Dtype> > >& bbox,
                   Blob<Dtype>* label_blob, Blob<Dtype>* bbox_blob);

  std::vector<std::shared_ptr<Blob<Dtype> > > subwin_data_out_blobs_;
  std::vector<Blob<Dtype>*> subwin_data_top_;
  int batch_size_;
  int subwin_data_cursor_;

  std::shared_ptr<bgm::AnnoEncoder<Dtype> > anno_encoder_;
  std::shared_ptr<bgm::AnnoDecoder<Dtype> > anno_decoder_;

};

// inline functions
template <typename Dtype>
inline SubwinData2Layer<Dtype>::SubwinData2Layer(const LayerParameter& param)
  : SubwinDataLayer<Dtype>(param) {

}

template <typename Dtype>
inline const char* SubwinData2Layer<Dtype>::type() const {
  return "SubwinData2";
}

template <typename Dtype>
inline int SubwinData2Layer<Dtype>::MinTopBlobs() const {
  return 1;
}

template <typename Dtype>
inline int SubwinData2Layer<Dtype>::MaxTopBlobs() const {
  return 3;
}

//template <typename Dtype>
//inline void SubwinData2Layer<Dtype>::ForwardWholeImg(
//    const Blob<Dtype>& src, Blob<Dtype>* dst) const {
//  dst->ShareData(src);
//}

} // namespace caffe

#endif // !TLR_SUBWIN_DATA_2_LAYER_HPP_
