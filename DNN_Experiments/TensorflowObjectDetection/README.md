Reference:

https://medium.com/@vijendra1125/custom-mask-rcnn-using-tensorflow-object-detection-api-101149ce0765

----------------------------------------------------------------------
run in models/research:

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

protoc object_detection/protos/*.proto --python_out=.




Python3 object_detection/dataset_tools/create_mask_rcnn_tf_record.py --data_dir_path=/Users/wangmeijie/ALLImportantProjects/FlameDetectionAPP/DNN_Experiments/TensorflowObjectDetection/dataset --annotations_dir=/Users/wangmeijie/ALLImportantProjects/FlameDetectionAPP/DNN_Experiments/TensorflowObjectDetection/dataset/Annotations --masks_dir=masks --xmls_dir=xmls --image_dir=/Users/wangmeijie/ALLImportantProjects/FlameDetectionAPP/DNN_Experiments/TensorflowObjectDetection/dataset/JPEGImages --label_map_path=//Users/wangmeijie/ALLImportantProjects/FlameDetectionAPP/DNN_Experiments/TensorflowObjectDetection/dataset/label.pbtxt --tfrecord_filename=train --use_xmls=False --num_shrads=1



python3 object_detection/legacy/train.py --train_dir=/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/CP --pipeline_config_path=/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/mask_rcnn_inception_v2_coco.config


python3 object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/mask_rcnn_inception_v2_coco.config --trained_checkpoint_prefix=/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/CP/model.ckpt-1371 --output_directory=/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/IG


