#ifndef BGM_BOX_HPP_
#define BGM_BOX_HPP_

#include <glog/logging.h>

#include <vector>

namespace bgm
{

template <typename T, class Allocator = std::allocator<T> >
class BoxContainer
{
 // nested classes
 public:
  class ConstIterator;
  class Iterator;
 private:
  class Shape
  {
   public:
    //Shape() {}
    Shape(const std::initializer_list<int>& shape = {0});

    int Count(int axis_begin = 0, int axis_end = -1) const;
    int Offset(const std::initializer_list<int>& idx) const;
    int Dimension() const;

    const std::vector<int>& shape() const;
    int shape(int axis) const;
    int step(int axis) const;

   private:
    std::vector<int> shape_;
    std::vector<int> step_;
  };


 public:
  BoxContainer() {}
//  BoxContainer(const std::initializer_list<int>& shape);
//  BoxContainer(const std::initializer_list<int>& shape, const T& value);
//
//  const T& At(const std::initializer_list<int>& idx) const;
//  T& At(const std::initializer_list<int>& idx);
//
//  int Offset(const std::initializer_list<int>& idx) const;
//  int Dimension() const;
//  int Count(int axis_begin = 0, int axis_end = -1) const;
//
//  //std::vector<T, Allocator>::const_iterator ConstDataBegin() const;
//  //std::vector<T, Allocator>::const_iterator ConstDataEnd() const;
//  //std::vector<T, Allocator>::iterator DataBegin();
//  //std::vector<T, Allocator>::iterator DataEnd();
//
//  //template <typename... Arg>
//  //ConstIterator GetConstIterator(Arg&&... args) const;
//  //template <typename... Arg>
//  //Iterator GetIterator(Arg&&... args) const;
//
// private:
//  std::vector<T, Allocator> data_;
//
  Shape shape_;
}; // class BoxContainer<T>

// nested classes
//template <typename T, class Allocator = std::allocator<T> >
//class BoxContainer<T, Allocator>::Shape
//{
 //public:
  //Shape(const std::initializer_list<int>& shape);

  //int Count(int axis_begin = 0, int axis_end = -1) const;
  //int Offset(const std::initializer_list<int>& idx) const;
  //int Dimension() const;

  //const std::vector<int>& shape() const;
  //int shape(int axis) const;
  //int step(int axis) const;

 //private:
 // std::vector<int> shape_;
 // std::vector<int> step_;
//};

//template <typename T, class Allocator = std::allocator<T> >
//class BoxContainer<T, Allocator>::ConstIterator
//{
// public:
//  ConstIterator();
//  ConstIterator(const BoxContainer<T, Allocator>::Shape& shape, 
//                const T* ptr, int direction = -1);
//  ConstIterator(const BoxContainer<T, Allocator>& box_container,
//                int offset = 0, int direction = -1);
//  ConstIterator(const BoxContainer<T, Allocator>& box_container,
//                const std::initializer_list<int>& idx,
//                int direction = -1);
//
//  const T& operator*() const;
//  const T* operator->() const;
//
//  ConstIterator& operator++();
//  ConstIterator operator++(int);    
//  ConstIterator& operator--();
//  ConstIterator operator--(int);
//
//  void set_direction(int direction);
//
// private:
//  int direction_;
//  
//  int step_;
//  const T* ptr_;
//
//  const BoxContainer<T, Allocator>::Shape& shape_;
//};

//template <typename T, class Allocator = std::allocator<T> >
//class BoxContainer<T, Allocator>::Iterator
//  : BoxContainer<T, Allocator>::ConstIterator
//{
// public:
//  T& operator*() const;
//  T* operator->() const;
//
//  Iterator& operator++();
//  Iterator operator++(int);    
//  Iterator& operator--();
//  Iterator operator--(int);
//};
//
//// template fucntions
//
//// BoxContainer template functions
//template <typename T, class Allocator = std::allocator<T> >
//inline BoxContainer<T, Allocator>::BoxContainer(
//    const std::initializer_list<int>& shape) : shape_(shape) {
//  int count = 1;
//  for (auto iter = size.begin(); iter != size.end(); ++iter)
//    count *= *iter;
//  data_.resize(count);
//  
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline BoxContainer<T, Allocator>::BoxContainer(
//    const std::initializer_list<int>& shape, const T& value) 
//  : BoxContainer(shape) {
//  std::fill(data_.begin(), data_.end(), value);
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline const T& BoxContainer<T, Allocator>::At(
//    const std::initializer_list<int>& idx) const {
//  return data_[shape_.Offset(idx)];
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline T& BoxContainer<T, Allocator>::At(
//    const std::initializer_list<int>& idx) {
//  return const_cast<T&>(static_cast<const BoxContainer<T, Allocator>*>(this)->At(idx));
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline int BoxContainer<T, Allocator>::Offset(
//    const std::initializer_list<int>& idx) const {
//  return shape_.Offset(idx);
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline int BoxContainer<T, Allocator>::Dimension() const {
//  return shape_.Dimension();
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline int BoxContainer<T, Allocator>::Count(int axis_begin, 
//                                             int axis_end) const {
//  return shape_.Count(axis_begin, axis_end);
//}
//
////template <typename T, class Allocator = std::allocator<T> >
////inline std::vector<T, Allocator>::const_iterator
////    BoxContainer<T, Allocator>::ConstDataBegin() const {
////  return data_.cbegin();
////}
////
////template <typename T, class Allocator = std::allocator<T> >
////inline std::vector<T, Allocator>::const_iterator
////    BoxContainer<T, Allocator>::ConstDataEnd() const {
////  return data_.cend();
////}
////
////template <typename T, class Allocator = std::allocator<T> >
////inline std::vector<T, Allocator>::iterator 
////    BoxContainer<T, Allocator>::DataBegin() {
////  return data_.begin();
////}
////
////template <typename T, class Allocator = std::allocator<T> >
////inline std::vector<T, Allocator>::iterator
////    BoxContainer<T, Allocator>::DataEnd() {
////  return data_.end();
////}
////
////template <typename T, class Allocator = std::allocator<T> >
////template <typename... Arg>
////inline BoxContainer<T, Allocator>::ConstIterator 
////BoxContainer<T, Allocator>::GetConstIterator(Arg&&... args) const {
////  return ConstIterator(*this, std::forward(args));
////}
////
////template <typename T, class Allocator = std::allocator<T> >
////template <typename... Arg>
////inline BoxContainer<T, Allocator>::Iterator 
////BoxContainer<T, Allocator>::GetIterator(Arg&&... args) const {
////  return Iterator(*this, std::forward(args));
////}
//
////template <typename T, class Allocator = std::allocator<T> >
////void BoxContainer<T, Allocator>::CheckIdx(
////    const std::initializer_list<int>& idx) const {
////  int num_idx = idx.size();
////  CHECK_GT(num_idx, 0);
////  CHECK_LT(num_idx, Dimension());
////  
////  auto iter = idx.begin()
////  for (int axis = 0; axis < num_idx; ++axis) {
////    int idx = *iter++;
////    CHECK_GE(idx, 0);
////    CHECK_LT(idx, shape_[axis]);
////  }
////}
////
////template <typename T, class Allocator = std::allocator<T> >
////inline int BoxContainer<T, Allocator>::Count(int axis_begin, int axis_end) const {
////  Count(shape_, axis_begin, axis_end);
////}
////
////template <typename T, class Allocator = std::allocator<T> >
////inline int BoxContainer<T, Allocator>::Count(
////    const std::vector<int> shape, int axis_begin, int axis_end) const {
////  int iter_end = (axis_end < 0) ? Dimension() : axis_end;
////    
////  int cnt = 1;
////  for (int i = axis_begin; i < iter_end; ++i)
////    cnt *= shape_[i];
////}
////
////template <typename T, class Allocator = std::allocator<T> >
////inline void BoxContainer<T, Allocator>::set_shape(
////    const std::initializer_list<int>& shape) {
////  //shape_.assign(shape.begin(), shape.end());
////  shape_ = shape;
////}
//
// Shape template functions
template <typename T, class Allocator = std::allocator<T> >
BoxContainer<T, Allocator>::Shape::Shape(
    const std::initializer_list<int>& shape) : shape_(shape) {
  int dim = shape.size();

  step_.resize(dim);

  step_[dim - 1] = shape_[0];
  for (int i = 1; i < dim; ++i)
    step_[dim - i - 1] = step_[dim - i] * shape_[i];
}

template <typename T, class Allocator = std::allocator<T> >
int BoxContainer<T, Allocator>::Shape::Offset(
    const std::initializer_list<int>& idx) const {
  int num_idx = idx.size();
    
  CHECK_GE(num_idx, 0);
  CHECK_LE(num_idx, shape_.size());
    
  auto idx_iter = idx.begin();
  int offset = 0;
  for (int i = 0; i < num_idx; ++i) {
    int current_idx = *idx_iter++;
    CHECK_GE(current_idx, 0);
    CHECK_LT(current_idx, shape_[i]);
    offset += step_[i] * current_idx;
  }
}

template <typename T, class Allocator = std::allocator<T> >
inline int BoxContainer<T, Allocator>::Shape::Count(
    int axis_begin, int axis_end) const {
  CHECK_GE(axis_begin, 0);
  CHECK_LE(axis_end, Dimension());
  int end = axis_end < 0 ? Dimension() : axis_end;
  CHECK_LT(axis_begin, end);

  int cnt = 1;
  for (int i = axis_begin; i < end; ++i)
    cnt *= shape_[i];

  return cnt;
}

template <typename T, class Allocator = std::allocator<T> >
inline int BoxContainer<T, Allocator>::Shape::Dimension() const {
  return shape_.size();
}

template <typename T, class Allocator = std::allocator<T> >
inline const std::vector<int>& 
BoxContainer<T, Allocator>::Shape::shape() const {
  return shape_;
}

template <typename T, class Allocator = std::allocator<T> >
inline int BoxContainer<T, Allocator>::Shape::shape(int axis) const {
  return shape_[axis];
}

template <typename T, class Allocator = std::allocator<T> >
inline int BoxContainer<T, Allocator>::Shape::step(int axis) const {
  return step_[axis];
}

//// ConstIterator template functions
//template <typename T, class Allocator = std::allocator<T> >
//BoxContainer<T, Allocator>::ConstIterator::ConstIterator() 
//  : direction_(0), step_(1), ptr_(nullptr) {
//
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//BoxContainer<T, Allocator>::ConstIterator::ConstIterator(
//    const BoxContainer<T, Allocator>::Shape& shape,
//    const T* ptr, int direction)
//  : shape_(shape), ptr_(ptr) {
//  CHECK(ptr);
//  set_direction(direction);
//  step_ = shape_.step(direction_);
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//BoxContainer<T, Allocator>::ConstIterator::ConstIterator(
//    const BoxContainer<T, Allocator>& box_container,
//    int offset, int direction) 
//  : ConstIterator(box_container.shape_, box_container.data_[offset],
//                  direction) 
//{
//
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//BoxContainer<T, Allocator>::ConstIterator::ConstIterator(
//    const BoxContainer<T, Allocator>& box_container,
//    const std::initializer_list<int>& idx, int direction) 
//  : ConstIterator(box_container.shape_, 
//                  box_container.shape_.Offset(idx),
//                  direction) 
//{
//
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline const T& 
//BoxContainer<T, Allocator>::ConstIterator::operator*() const {
//  return *ptr_;
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline const T* 
//BoxContainer<T, Allocator>::ConstIterator::operator->() const {
//  return ptr_;
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline BoxContainer<T, Allocator>::ConstIterator& 
//BoxContainer<T, Allocator>::ConstIterator::operator++() {
//  if(step_ == 1) ++ptr_;
//  else ptr += step;
//  return (*this);
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline BoxContainer<T, Allocator>::ConstIterator 
//BoxContainer<T, Allocator>::ConstIterator::operator++(int) {
//  ConstIterator temp(*this);
//  ++(*this);
//  return temp;
//}
//    
//template <typename T, class Allocator = std::allocator<T> >
//inline BoxContainer<T, Allocator>::ConstIterator& 
//BoxContainer<T, Allocator>::ConstIterator::operator--() {
//  if(step_ == 1) --ptr_;
//  else ptr -= step;
//  return (*this);
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline BoxContainer<T, Allocator>::ConstIterator 
//BoxContainer<T, Allocator>::ConstIterator::operator--(int) {
//  ConstIterator temp(*this);
//  --(*this);
//  return temp;
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline void 
//BoxContainer<T, Allocator>::ConstIterator::set_direction(int direction) {
//  direction_ = std::max(0, direction);
//  CHECK_LT(direction_, shape_.Dimension());
//}
//
//// Iterator template functions
//template <typename T, class Allocator = std::allocator<T> >
//inline T& BoxContainer<T, Allocator>::Iterator::operator*() const {
//  return const_cast<T&>(ConstIterator::operator*());
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline T* BoxContainer<T, Allocator>::Iterator::operator->() const {
//  return const_cast<T*>(ConstIterator::operator->());
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline BoxContainer<T, Allocator>::Iterator& 
//BoxContainer<T, Allocator>::Iterator::operator++() {
//  ++(reinterpret_cast<ConstIterator*>(this));
//  return (*this);
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline BoxContainer<T, Allocator>::Iterator 
//BoxContainer<T, Allocator>::Iterator::operator++(int) {
//  Iterator temp = *this;
//  ++(*this);
//  return temp;
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline BoxContainer<T, Allocator>::Iterator& 
//BoxContainer<T, Allocator>::Iterator::operator--() {
//  --(reinterpret_cast<ConstIterator*>(this));
//  return (*this);
//}
//
//template <typename T, class Allocator = std::allocator<T> >
//inline BoxContainer<T, Allocator>::Iterator 
//BoxContainer<T, Allocator>::Iterator::operator--(int) {
//  Iterator temp = *this;
//  --(*this);
//  return temp;
//}

} // namespace bgm

#endif // !BGM_BOX_HPP_