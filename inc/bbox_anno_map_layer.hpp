#ifndef TRL_BBOX_ANNO_MAP_LAYER_HPP_
#define TRL_BBOX_ANNO_MAP_LAYER_HPP_

#include "bbox.hpp"

#include "caffe/layer.hpp"

namespace caffe
{

//template <typename Dtype>
//struct BBoxAnno
//{
//  Dtype label;
//  Dtype x_min;
//  Dtype y_min;
//  Dtype x_max;
//  Dtype y_max;
//
//  BBoxAnno();
//  BBoxAnno(Dtype label, 
//           Dtype x_min, Dtype y_min,
//           Dtype x_max, Dtype y_max);
//};

template <typename Dtype>
class BBoxAnnoMapLayer : public Layer<Dtype>
{
  typedef std::pair<Dtype, bgm::BBox<Dtype> > BBoxAnno;

  public:
    explicit BBoxAnnoMapLayer(const LayerParameter& param);
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;
    virtual void Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) override;

    virtual const char* type() const override;

    /**
    bottom is ImgBBoxLayer's label output
    return : 1
    */
    virtual int ExactNumBottomBlobs() const override;

    /**
    There are two tops. label map and bbox map
    return : 2
    */
    virtual int ExactNumTopBlobs() const override;

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
    /**
    Initialize output shapes
    */
    void InitMapShape(const LabelParameter& label_param,
                      const BBoxAnnoMapParameter& bbox_anno_map_param);

    /**
    compute output's height and width
    */
    void ComputeMapHW(int *map_height, int *map_width) const;

    void MakeMaps(const vector<BBoxAnno>& bbox_anno,
                  int map_height, int map_width,
                  Dtype* label_map,
                  Dtype* bbox_map) const;

    /**
    find best BBoxAnno for given feceptive_field.
    return its index or -1 for failure.
    */
    int FindBestBBoxAnno(
        const bgm::BBox<Dtype>& receptive_field,
        const vector<BBoxAnno>& candidates) const;

    void ParseInputBlob(
        const Blob<Dtype>& input_blob,
        vector<vector<BBoxAnno > >* bbox_anno) const;
    
    bool IsBBoxInReceptiveField(const bgm::BBox<Dtype>& receptive_field,
                                const bgm::BBox<Dtype>& obj_bbox) const;
    float ComputeCenterDistance(const bgm::BBox<Dtype>& bbox1,
                                const bgm::BBox<Dtype>& bbox2) const;

    void RelocateBBox(const bgm::BBox<Dtype>& receptive_field,
                      const bgm::BBox<Dtype>& global_position,
                      bgm::BBox<Dtype>* local_position) const;

    const int NUM_LABEL_;
    const int IMG_HEIGHT_;
    const int IMG_WIDTH_;
    const int RECEPTIVE_FIELD_HEIGHT_;
    const int RECEPTIVE_FIELD_WIDTH_;
    const int VERTICAL_STRIDE_;
    const int HORIZONTAL_STRIDE_;
    const bool NORMALIZED_POSITION_IN_;
    const bool NORMALIZED_POSITION_OUT_;

    vector<int> labelmap_shape_;
    vector<int> bboxmap_shape_;
};

// inline functions
//template <typename Dtype>
//inline BBoxAnno<Dtype>::BBoxAnno() 
//  : label(LabelParameter::DUMMY_LABEL) { }
//
//template <typename Dtype>
//inline BBoxAnno<Dtype>::BBoxAnno(Dtype label, 
//                                 Dtype x_min, Dtype y_min,
//                                 Dtype x_max, Dtype y_max) 
//  : label(label), x_min(x_min), y_min(y_min),
//    x_max(x_max), y_max(y_max) { 
//  CHECK(label >= 0);
//}

template <typename Dtype>
inline void BBoxAnnoMapLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  InitMapShape(this->layer_param().label_param(),
               this->layer_param().bbox_anno_map_param());
}

template <typename Dtype>
inline const char* BBoxAnnoMapLayer<Dtype>::type() const {
  return "BBoxAnnoMap";
}

template <typename Dtype>
inline int BBoxAnnoMapLayer<Dtype>::ExactNumBottomBlobs() const {
  return 1;
}

template <typename Dtype>
inline int BBoxAnnoMapLayer<Dtype>::ExactNumTopBlobs() const {
  return 2;
}

template <typename Dtype>
inline void BBoxAnnoMapLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}

template <typename Dtype>
inline void BBoxAnnoMapLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}

} // namespace caffe

#endif // !TRL_BBOX_ANNO_MAP_LAYER_HPP_
