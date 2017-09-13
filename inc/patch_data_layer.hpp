#ifndef TLR_PATCH_DATA_LAYER_HPP_
#define TLR_PATCH_DATA_LAYER_HPP_

#include "caffe/layers/data_layer.hpp"

namespace caffe
{

template <typename Dtype>
class PatchDataLayer : public DataLayer<Dtype>
{
  public:
    explicit PatchDataLayer(const LayerParameter& param);
    virtual void DataLayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual const char* type() const override;
    virtual int MaxTopBlobs() const override;

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual void load_batch(caffe::Batch<Dtype>* batch) override;

  private:
    void PrepareCopy(const PatchDatum& datum, Batch<Dtype>* batch);
    void CopyImage(int item_id, const PatchDatum& datum, 
                   Blob<Dtype>* batch_data);
    void CopyLabel(int item_id,
                   const PatchDatum& datum,
                   Blob<Dtype>* batch_label) const;

    void GetPatchOffset(const PatchDatum& datum, Dtype* dst) const;
    void GetBBox(const PatchDatum& datum, Dtype* dst) const;

    void ExtractLabelOut(const Blob<Dtype>& prefetched_label,
                         Blob<Dtype>* label) const;
    void ExtractPatchOffsetOut(const Blob<Dtype>& prefetched_label,
                               Blob<Dtype>* patch_offset) const;
    void ExtractBBoxOut(const Blob<Dtype>& prefetched_label,
                        Blob<Dtype>* bbox) const;

    const int BATCH_SIZE_;
    const int NUM_LABEL_;
    const bool POSITIVE_ONLY_;
    const bool PATCH_OFFSET_NORMALIZATION_;
    const bool BBOX_NORMALIZATION_;

    Blob<Dtype> label_;
    Blob<Dtype> patch_offset_;
    Blob<Dtype> bbox_;
};

// inline functions
template <typename Dtype>
inline const char* PatchDataLayer<Dtype>::type() const {
  return "PatchData";
}

template <typename Dtype>
inline int PatchDataLayer<Dtype>::MaxTopBlobs() const {
  return 4;
}

template <typename Dtype>
inline void PatchDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

} // namespace caffe

#endif // !TLR_PATCH_DATA_LAYER_HPP_