#ifndef TLR_BBOX_HPP_
#define TLR_BBOX_HPP_

#include <assert.h>
#include <vector>

namespace bgm
{

template <typename Dtype>
class BBox
{
  enum Idx {X_MIN, Y_MIN, X_MAX, Y_MAX};

  public:
    BBox();
    BBox(const Dtype& x_min,
         const Dtype& y_min,
         const Dtype& x_max,
         const Dtype& y_max);
    ~BBox();

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
    void set_x_min(const Dtype& value) const;
    void set_y_min(const Dtype& value) const;
    void set_x_max(const Dtype& value) const;
    void set_y_max(const Dtype& value) const;

  private:
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
  : bbox_({x_min, y_min, x_max, y_max}) {
}

template <typename Dtype>
inline BBox<Dtype>::~BBox() {
  delete bbox_[0];
  delete bbox_[1];
  delete bbox_[2];
  delete bbox_[3];
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
inline void BBox<Dtype>::set_x_min(const Dtype& value) const {
  bbox_[X_MIN] = value;
}

template <typename Dtype>
inline void BBox<Dtype>::set_y_min(const Dtype& value) const{
  bbox_[Y_MIN] = value;
}

template <typename Dtype>
inline void BBox<Dtype>::set_x_max(const Dtype& value) const{
  bbox_[X_MAX] = value;
}

template <typename Dtype>
inline void BBox<Dtype>::set_y_max(const Dtype& value) const {
  bbox_[Y_MAX] = value;
}


} // namespace bgm

#endif // !TLR_BBOX_HPP_