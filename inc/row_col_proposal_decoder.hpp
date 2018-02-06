#ifndef BGM_ROW_COL_PROPOSAL_DECODER_HPP_
#define BGM_ROW_COL_PROPOSAL_DECODER_HPP_

#include "proposal_decoder.hpp"

namespace bgm
{

class RowColProposalDecoder : public ProposalDecoder
{
 public:
  RowColProposalDecoder(int row, int col,
                        int img_width = 0, int img_height = 0);

#ifdef USE_CAFFE
  template <typename Dtype>
  virtual void Decode(
      const caffe::Blob<Dtype>& proposal,
      std::vector<std::vector<cv::Rect_<Dtype> > >* decoded);
#endif // USE_CAFFE

 private:
  int rows_;
  int cols_;
};

// inline functions

// template functions

} // namespace bgm

#endif // BGM_ROW_COL_PROPOSAL_DECODER_HPP_
