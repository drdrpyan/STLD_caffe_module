#include "iou_layer.hpp"

namespace caffe
{

template <typename Dtype>
void IOULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& bbox = *(bottom[0]);
  const Blob<Dtype>& bbox_gt = *(bottom[1]);
  const Blob<Dtype>& label_gt = *(bottom[2]);

  CHECK(bbox.shape() == bbox_gt.shape());
  CHECK_EQ(bbox.num(), label_gt.num());
  CHECK_EQ(bbox.height(), label_gt.height());
  CHECK_EQ(bbox.width(), label_gt.width());
  
  std::vector<int> out_shape(0); 
  top[0]->Reshape(out_shape);
}

template <typename Dtype>
void IOULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& bbox = *(bottom[0]);
  const Blob<Dtype>& bbox_gt = *(bottom[1]);
  const Blob<Dtype>& label_gt = *(bottom[2]);

  Dtype avg_iou = 0;
  int obj_count = 0;

  for (int n = 0; n < bbox.num(); ++n) {
    const Dtype* bbox_x_iter = bbox.cpu_data() + bbox.offset(n, 0);
    const Dtype* bbox_y_iter = bbox.cpu_data() + bbox.offset(n, 1);
    const Dtype* bbox_w_iter = bbox.cpu_data() + bbox.offset(n, 2);
    const Dtype* bbox_h_iter = bbox.cpu_data() + bbox.offset(n, 3);

    const Dtype* gt_x_iter = bbox_gt.cpu_data() + bbox_gt.offset(n, 0);
    const Dtype* gt_y_iter = bbox_gt.cpu_data() + bbox_gt.offset(n, 1);
    const Dtype* gt_w_iter = bbox_gt.cpu_data() + bbox_gt.offset(n, 2);
    const Dtype* gt_h_iter = bbox_gt.cpu_data() + bbox_gt.offset(n, 3);

    const Dtype* label_iter = label_gt.cpu_data() + label_gt.offset(n);

    for (int h = 0; h < bbox.height(); ++h) {
      for (int w = 0; w < bbox.width(); ++w) {
        if (*label_iter != LabelParameter::NONE &&
            *label_iter != LabelParameter::DUMMY_LABEL) {
          obj_count++;

          avg_iou += ComputIOU(*bbox_x_iter, *bbox_y_iter,
                               *bbox_w_iter, *bbox_h_iter,
                               *gt_x_iter, *gt_y_iter,
                               *gt_w_iter, *gt_h_iter);
        }
        
        bbox_x_iter++;
        bbox_y_iter++;
        bbox_w_iter++;
        bbox_h_iter++;

        gt_x_iter++;
        gt_y_iter++;
        gt_w_iter++;
        gt_h_iter++;

        label_iter++;
      }
    }

  }

  if (obj_count)
    avg_iou /= obj_count;

  top[0]->mutable_cpu_data()[0] = avg_iou;
  
}

template <typename Dtype>
Dtype IOULayer<Dtype>::ComputIOU(Dtype x, Dtype y, Dtype w, Dtype h,
                                 Dtype gt_x, Dtype gt_y,
                                 Dtype gt_w, Dtype gt_h) const {
  Dtype w_begin = std::max(x - w / 2., gt_x - gt_w / 2.);
  Dtype w_end = std::min(x + w / 2., gt_x + gt_w / 2.);
  Dtype w_overlap = (w_begin < w_end) ? w_end - w_begin : 0;

  Dtype h_begin = std::max(y - h / 2., gt_y - gt_h / 2.);
  Dtype h_end = std::min(y + h / 2., gt_y + gt_h / 2.);
  Dtype h_overlap = (h_begin < h_end) ? h_end - h_begin : 0;

  Dtype inter_area = w_overlap * h_overlap;
  Dtype union_area = w*h + gt_w*gt_h - inter_area;

  return inter_area / union_area;
}

#ifdef CPU_ONLY
STUB_GPU(IOULayer);
#endif

INSTANTIATE_CLASS(IOULayer);
REGISTER_LAYER_CLASS(IOU);

}