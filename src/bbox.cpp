#include "bbox.hpp"

namespace bgm
{

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
      Dtype mid = ((*min) + (*max)) / static_cast<Dtype>(2);
      *min = ((*min) - mid) * scale_factor + mid;
      *max = ((*max) - mid) * scale_factor + mid;
      break;
    default:
      assert(flase);
      break;
  }
}

} // namespace bgm