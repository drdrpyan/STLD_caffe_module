#ifndef BGM_PROPOSAL_DECODER_HPP_
#define BGM_PROPOSAL_DECODER_HPP_


#ifdef USE_CAFFE
#include "caffe/blob.hpp"  
#endif // USE_CAFFE

#include <opencv2/core.hpp>

#include <vector>

namespace bgm
{

class ProposalDecoder
{
 public:
#ifdef USE_CAFFE
  template <typename Dtype>
  virtual void Decode(
      const caffe::Blob<Dtype>& proposal,
      std::vector<std::vector<cv::Rect_<Dtype> > >* decoded) = 0;
#endif // USE_CAFFE
};

} // namespace bgm

#endif// !BGM_PROPOSAL_DECODER_HPP_
