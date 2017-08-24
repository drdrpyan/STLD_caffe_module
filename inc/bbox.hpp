#ifndef TLR_BBOX_HPP_
#define TLR_BBOX_HPP_

#include <assert.h>
#include <vector>

namespace bgm
{

template <typename Dtype>
class BBox
{
  public:
    enum ScalePivot {SCENE_TOPLEFT, 
                     BBOX_TOPLEFT,
                     BBOX_CENTER};
  private:
    enum Idx { X_MIN, Y_MIN, X_MAX, Y_MAX };

  public:
    BBox();
    BBox(const Dtype& x_min,
         const Dtype& y_min,
         const Dtype& x_max,
         const Dtype& y_max);
    BBox(const BBox& ref);
    ~BBox();

    void Shift(const Dtype& shift_x, const Dtype& shift_y);
    void ShiftX(const Dtype& shift_dist);
    void ShiftY(const Dtype& shift_dist);
    void Scale(const Dtype& scale_x, const Dtype& scale_y,
               ScalePivot pivot);
    void ScaleX(const Dtype& scale_x, ScalePivot pivot);
    void ScaleY(const Dtype& scale_y, ScalePivot pivot);

    const Dtype* Get() const;
    void Get(std::vector<Dtype>* vec) const;
    void Set(const Dtype& x_min,
             const Dtype& y_min,
             const Dtype& x_max,
             const Dtype& y_max);
    void Set(const Dtype* arr);
    void Set(const std::vector<Dtype>& vec);

    const Dtype& x_min() const;
    const Dtype& y_min() const;
    const Dtype& x_max() const;
    const Dtype& y_max() const;
    void set_x_min(const Dtype& value);
    void set_y_min(const Dtype& value);
    void set_x_max(const Dtype& value);
    void set_y_max(const Dtype& value);

    BBox& operator=(const BBox& rhs);

  private:
    void Scale(const Dtype& scale_factor,
               ScalePivot pivot,
               Dtype* min, Dtype* max);

    Dtype bbox_[4];
};

// inline functions
template <typename Dtype>
inline BBox<Dtype>::BBox() {
  // do nothing
}

template <typename Dtype>
inline BBox<Dtype>::BBox(const Dtype& x_min,
                         const Dtype& y_min,
                         const Dtype& x_max,
                         const Dtype& y_max) 
  /*: bbox_({x_min, y_min, x_max, y_max})*/ {
  bbox_[X_MIN] = x_min;
  bbox_[Y_MIN] = y_min;
  bbox_[X_MAX] = x_max;
  bbox_[Y_MAX] = y_max;
}

template <typename Dtype>
inline BBox<Dtype>::BBox(const BBox& ref) {
  std::copy(ref.bbox_, ref.bbox_ + sizeof(Dtype) * 4, bbox_);
}

template <typename Dtype>
inline BBox<Dtype>::~BBox() {
  //delete bbox_[0];
  //delete bbox_[1];
  //delete bbox_[2];
  //delete bbox_[3];
}

template <typename Dtype>
inline void BBox<Dtype>::Shift(const Dtype& shift_x,
                               const Dtype& shift_y) {
  ShiftX(shift_x);
  ShiftY(shift_y);
}

template <typename Dtype>
inline void BBox<Dtype>::ShiftX(const Dtype& shift_dist) {
  bbox_[X_MIN] += shift_dist;
  bbox_[X_MAX] += shift_dist;
}
    
template <typename Dtype>
inline void BBox<Dtype>::ShiftY(const Dtype& shift_dist) {
  bbox_[Y_MIN] += shift_dist;
  bbox_[Y_MAX] += shift_dist;
}

template <typename Dtype>
inline void BBox<Dtype>::Scale(const Dtype& scale_x,
                               const Dtype& scale_y,
                               ScalePivot pivot) {
  ScaleX(scale_x, pivot);
  ScaleY(scale_y, pivot);
}

template <typename Dtype>
inline void BBox<Dtype>::ScaleX(const Dtype& scale_x, 
                                ScalePivot pivot) {
  Scale(scale_x, pivot, bbox_ + X_MIN, bbox_ + X_MAX);
}

template <typename Dtype>
inline void BBox<Dtype>::ScaleY(const Dtype& scale_y, 
                                ScalePivot pivot) {
  Scale(scale_y, pivot, bbox_ + Y_MIN, bbox_ + Y_MAX);
}

template <typename Dtype>
inline const Dtype* BBox<Dtype>::Get() const {
  return bbox_;
}

template <typename Dtype>
inline void BBox<Dtype>::Get(std::vector<Dtype>* vec) const {
  assert(vec);
  vec->assign(bbox_, bbox_ + 3);
}

template <typename Dtype>
inline void BBox<Dtype>::Set(const Dtype& x_min,
                             const Dtype& y_min,
                             const Dtype& x_max,
                             const Dtype& y_max) {
  bbox_[X_MIN] = x_min;
  bbox_[Y_MIN] = y_min;
  bbox_[X_MAX] = x_max;
  bbox_[Y_MAX] = y_max;
}

template <typename Dtype>
inline void BBox<Dtype>::Set(const Dtype* arr) {
  assert(arr);
  std::copy(arr, arr + 4, bbox_);
}

template <typename Dtype>
inline void BBox<Dtype>::Set(const std::vector<Dtype>& vec) {
  assert(vec.size() == 4);
  std::copy(vec.cbegin(), vec.cend(), bbox_);
}
template <typename Dtype>
inline const Dtype& BBox<Dtype>::x_min() const {
  return bbox_[X_MIN];
}

template <typename Dtype>
inline const Dtype& BBox<Dtype>::y_min() const {
  return bbox_[Y_MIN];
}

template <typename Dtype>
inline const Dtype& BBox<Dtype>::x_max() const {
  return bbox_[X_MAX];
}

template <typename Dtype>
inline const Dtype& BBox<Dtype>::y_max() const {
  return bbox_[Y_MAX];
}

template <typename Dtype>
inline void BBox<Dtype>::set_x_min(const Dtype& value) {
  bbox_[X_MIN] = value;
}

template <typename Dtype>
inline void BBox<Dtype>::set_y_min(const Dtype& value) {
  bbox_[Y_MIN] = value;
}

template <typename Dtype>
inline void BBox<Dtype>::set_x_max(const Dtype& value) {
  bbox_[X_MAX] = value;
}

template <typename Dtype>
inline void BBox<Dtype>::set_y_max(const Dtype& value) {
  bbox_[Y_MAX] = value;
}

template <typename Dtype>
inline BBox<Dtype>& BBox<Dtype>::operator=(const BBox<Dtype>& rhs) {
  std::copy(rhs.bbox_, rhs.bbox_ + sizeof(Dtype) * 4, bbox_);
  return *this;
}

template <typename Dtype>
void BBox<Dtype>::Scale(const Dtype& scale_factor,
                        ScalePivot pivot,
                        Dtype* min, Dtype* max) {
  assert(min && max);

  switch (pivot) {
    case SCENE_TOPLEFT:
      (*min) *= scale_factor;
      (*max) *= scale_factor;
      break;
    case BBOX_TOPLEFT:
      (*max) *= scale_factor;
      break;
    case BBOX_CENTER:
      {
        Dtype mid = ((*min) + (*max)) / static_cast<Dtype>(2);
        *min = ((*min) - mid) * scale_factor + mid;
        *max = ((*max) - mid) * scale_factor + mid;
      }
      break;
    default:
      assert(false);
      break;
  }
}
} // namespace bgm

#endif // !TLR_BBOX_HPP_