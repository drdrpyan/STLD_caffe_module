# caffe module for Small Traffic Light Detector (STLD)

Sources for small traffic light detector which are implemented with caffe framework.

My STLD is YOLOv2 based encoder-decoder extension, and it uses focal regression loss (inspired by focal loss) to regress confidences of bounding boxes.

## directory

root

	|--inc: headers for STLD

	|--include: headers for caffe (slightly modified)

	|--proto: prototxt for caffe and its corresponding source files

	|--src: sources for STLD

		|--caffe: sources for caffe (slightly modified)