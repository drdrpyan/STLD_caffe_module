#ifndef TLR_PATCH_INFO_HANDLER_HPP_
#define TLR_PATCH_INFO_HANDLER_HPP_

#include "glog/logging.h"
#include "caffe/blob.hpp"

namespace caffe
{

template <typename Dtype>
class PatchInfoHandler
{
  public:
    PatchInfoHandler();
    void GetOffsets(Blob<Dtype>* offsets) const;
    void ResolutionAugment(const Blob<Dtype>& label,
                           const Blob<Dtype>& bbox,
                           Blob<Dtype>* augmented_label,
                           Blob<Dtype>* augmented_bbox) const;
    void Decode(const Blob<Dtype>)



    void set_whole_img_width(int whole_img_width);
    void set_whole_img_height(int whole_img_height);
    void set_patch_width(int patch_width);
    void set_patch_height(int patch_height);
    void set_horizontal_stride(int horizontal_stride);
    void set_vertical_stride(int vertical_stride);
    void set_offset_normalization(bool normalization);
    void set_bbox_normalization(bool normalization);

    static void GetPatchOffsets(int whole_img_width, int whole_img_height,
                                int patch_width, int patch_height,
                                int horizontal_stride, int vertical_stride,
                                bool offset_normalization,
                                bool bbox_normalization,
                                Blob<Dtype>* offsets);
    static void ResolutionAugment(const Blob<Dtype>& label,
                                  const Blob<Dtype>& bbox,
                                  int whole_img_width, int whole_img_height,
                                  int patch_width, int patch_height,
                                  int horizontal_stride, int vertical_stride,
                                  int horizontal_augmentation,
                                  int vertical_augmentation,
                                  bool offset_normalization,
                                  bool bbox_normalization,
                                  Blob<Dtype>* augmented_label,
                                  Blob<Dtype>* augmented_bbox);
    //static void DecodePrediction

  private:
    int whole_img_width_;
    int whole_img_height_;
    int patch_width_;
    int patch_height_;
    int horizontal_stride_;
    int vertical_stride_;
    bool offset_normalization_;
    bool bbox_normalization_;
};

// inline functions
template <typename Dtype>
inline void PatchInfoHandler<Dtype>::set_whole_img_width(int whole_img_width) {
  CHECK_GT(whole_img_width, 0);
  while_img_width_ = whole_img_width;
}

template <typename Dtype>
inline void PatchInfoHandler<Dtype>::set_whole_img_height(int whole_img_height) {
  CHECK_GT(whole_img_height, 0);
  while_img_height_ = whole_img_height;
}

template <typename Dtype>
inline void PatchInfoHandler<Dtype>::set_patch_width(int patch_width) {
  CHECK_GT(patch_width, 0);
  patch_width_ = patch_width;
}

template <typename Dtype>
inline void PatchInfoHandler<Dtype>::set_patch_height(int patch_height) {
  CHECK_GT(patch_height, 0);
  patch_height_ = patch_height;
}

template <typename Dtype>
inline void PatchInfoHandler<Dtype>::set_horizontal_stride(int horizontal_stride) {
  CHECK_GT(horizontal_stride, 0);
  horizontal_stride_ = horizontal_stride;
}

template <typename Dtype>
inline void PatchInfoHandler<Dtype>::set_vertical_stride(int vertical_stride) {
  CHECK_GT(vertical_stride, 0);
  vertical_stride_ = vertical_stride;
}

template <typename Dtype>
inline void PatchInfoHandler<Dtype>::set_offset_normalization(bool normalization) {
  offset_normalization_ = normalization;
}

template <typename Dtype>
inline void PatchInfoHandler<Dtype>::set_bbox_normalization(bool normalization) {
  offset_normalization_ = normalization;
}

} // namespace caffe

#endif // !TLR_PATCH_INFO_HANDLER_HPP_
