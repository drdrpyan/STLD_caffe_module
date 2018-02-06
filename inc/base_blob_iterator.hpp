//#ifndef TLR_BASE_BLOB_ITERATOR_HPP_
//#define TLR_BASE_BLOB_ITERATOR_HPP_
//
//#include <vector>
//
//#include "caffe/blob.hpp"
//#include "caffe/common.hpp"
//#include "caffe/util/math_functions.hpp"
//
//namespace caffe
//{
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//class BaseBlobIterator
//{
//  public:
//    enum Axis {N = 0, C, H, W};
//
//  public:
//    void Copy(int count, std::vector<Dtype>* dst) const;    
//    void Copy(int count, Axis axis, std::vector<Dtype>* dst) const;  
//    void Copy(int count, Dtype* dst) const;
//    void Copy(int count, Axis axis, Dtype* dst) const;      
//    
//    BaseBlobIterator& Jump(int jump);
//    BaseBlobIterator& Jump(Axis axis, int jump);
//
//    Axis axis() const;
//    void set_axis(Axis axis);
//
//    Dtype operator*() const;
//    const Dtype* operator->() const;
//
//    BaseBlobIterator& operator++();
//    const BaseBlobIterator& operator++(int);
//    BaseBlobIterator& operator--();
//    const BaseBlobIterator& operator--(int);
//
//    const BaseBlobIterator& operator+(int jump);
//    BaseBlobIterator& operator+=(int jump);
//    const BaseBlobIterator& operator-(int jump);
//    BaseBlobIterator operator-=(int jump);
//
//    //bool operator<=(const BaseBlobIterator& lhs) const;
//    //bool operator<(const BaseBlobIterator& lhs) const;
//    //bool operator>(const BaseBlobIterator& lhs) const;
//    //bool operator>=(const BaseBlobIterator& lhs) const;
//    //bool operator==(const BaseBlobIterator& lhs) const;
//    //bool operator!=(const BaseBlobIterator& lhs) const;
//
//  protected:
//    BaseBlobIterator(BlobType& blob, int offset = 0, Axis axis = N);
//
//  private:
//    virtual PtrType GetBlobBegin(BlobType& blob) const = 0;
//    bool CheckBound() const;
//    bool CheckBound(PtrType target) const;
//    int GetStride(Axis axis) const;
//
//    PtrType begin_;
//    PtrType end_;
//    std::vector<int> shape_;
//    
//    Axis axis_;
//    int stride_;
//
//    PtrType current_;
//};
//
//// inline functions
//template <typename Dtype, typename PtrType, typename BlobType, 
//          Caffe::Brew Device>
//inline void BaseBlobIterator<Dtype, PtrType, BlobType, Device>::Copy(
//    int count, std::vector<Dtype>* dst) const {
//  Copy(count, axis_, dst);
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, 
//          Caffe::Brew Device>
//inline void BaseBlobIterator<Dtype, PtrType, BlobType, Device>::Copy(
//    int count, Axis axis, std::vector<Dtype>* dst) const {
//  CHECK_GT(count, 0);
//  CHECK(dst);
//  dst->resize(count);
//  Copy(count, axis_, &(dst[0]));
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//inline void BaseBlobIterator<Dtype, PtrType, BlobType, Device>::Copy(
//    int count, Dtype* dst) const {
//  Copy(count, axis_, dst);
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//inline void BaseBlobIterator<Dtype, PtrType, BlobType, Device>::Copy(
//    int count, Dtype* dst) const {
//  Copy(count, axis_, dst);
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//inline void BaseBlobIterator<Dtype, PtrType, BlobType, Device>::Copy(
//    int count, Dtype* dst) const {
//  Copy(count, axis_, dst);
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//inline BaseBlobIterator<Dtype, PtrType, BlobType, Device>& 
//    BaseBlobIterator<Dtype, PtrType, BlobType, Device>::Jump(int jump) {
//  return Jump(axis_, jump);
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//inline BaseBlobIterator<Dtype, PtrType, BlobType, Device>::Axis 
//    BaseBlobIterator<Dtype, PtrType, BlobType, Device>::axis() const {
//  return axis_;
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//inline void BaseBlobIterator<Dtype, PtrType, BlobType, Device>::set_axis(Axis axis) {
//  axis_ = axis;
//  stride_ = GetStride(axis);
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//BaseBlobIterator<Dtype, PtrType, BlobType, Device>::PtrType() const {
//  return static_cast<PtrType>(current_);
//}
//
//
//
//
//
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//inline BaseBlobIterator<Dtype, PtrType, BlobType, Device>::BaseBlobIterator(
//    BlobType& blob, int offset, Axis axis) 
//  : begin_(GetBlobBegin(blob)), shape_(blob.shape()) {
//  end_ = begin_ + blob.count();
//  set_axis(axis);
//  current_ = begin_ + offset;
//
//  CheckBound();
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//inline bool BaseBlobIterator<Dtype, PtrType, BlobType, Device>::CheckBound() const {
//  CheckBound(current_);
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//inline bool BaseBlobIterator<Dtype, PtrType, BlobType, Device>::CheckBound(
//    PtrType target) const {
//  CHECK_GE(target, begin_) << "Target pointer is out of Blob's boundary";
//  CHECK_LT(target, end_) << "Target pointer is out of Blob's boundary";
//  return true;
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//inline int BaseBlobIterator<Dtype, PtrType, BlobType, Device>::GetStride(
//    Axis axis) const {
//  int stride = 1;
//  for (int i = static_cast<int>(axis) + 1; i < 4; i++)
//    stride *= shape_[i];
//  return stride;
//}
//
//// template functions
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//inline void BaseBlobIterator<Dtype, PtrType, BlobType, Device>::Copy(
//    int count, Axis axis, Dtype* dst) const {
//  CHECK(CheckBound(current_ + count * stride_));
//  CHECK(dst);
//
//  int stride = (axis == axis_) ? stride_ : GetStride(axis);
//
//  if (stride == 1)
//    caffe_copy(count, current_, dst);
//  else {
//    PtrType src_iter = current_;
//    Dtype* dst_iter = dst;
//    for (int i = count; i--; ) {
//      *dst_iter++ = *src_iter;
//      src_iter += stride;
//    }
//  }
//}
//
//template <typename Dtype, typename PtrType, typename BlobType, Caffe::Brew Device>
//BaseBlobIterator<Dtype, PtrType, BlobType, Device>& 
//    BaseBlobIterator<Dtype, PtrType, BlobType, Device>::Jump(Axis axis, int jump) {
//  int stride = (axis == axis_) ? stride_ : GetStride(axis);
//  PtrType target = current_ + stride * jump;
//  CheckBound(target);
//  current_ = target;
//  return *this;
//}
//
//// instantiate
//template <typename Dtype, typename PtrType, typename BlobType>
//typedef BaseBlobIterator<Dtype, PtrType, BlobType, Caffe::CPU>
//    BaseBlobCPUIterator<Dtype, PtrType, BlobType>;
//
//template <typename Dtype, typename PtrType, typename BlobType>
//typedef BaseBlobIterator<Dtype, PtrType, BlobType, Caffe::GPU>
//    BaseBlobGPUIterator<Dtype, PtrType, BlobType>;
//
//} // namespace caffe
//#endif // !TLR_BASE_BLOB_ITERATOR_HPP_
