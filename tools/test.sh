python ./inference_image.py --config-file config_path  --weight-file weight_path  --image test_images_list_txt --save_dir=save_dir --device device_id --start_imgid start_imgid

#a example modify project_path as your FCOS dir path
python ./inference_image.py --config-file 'project_pat/configs/rfcos/rfcos_R_101_FPN_2x_fovea_range_focalloss_alpha.yaml'  --weight-file 'project_path/training_dir/115_dota_nosqrt3/model_0080000.pth'   --image ./data/dota2018test.txt --save_dir="115_dota_nosqrt3" --device ${1} --start_imgid ${2}
