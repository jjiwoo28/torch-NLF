stanford_path="/data/hmjung/data_backup/NeuLF/dataset/stanford_half"
llff_path="/data/hmjung/data_backup/gaussian-splatting/dataset/nerf_llff_data"
bmw_path="data"
bmw1="bike_1by1_direction1_new"


llff1="fern"
stanford1="knights"
stanford2="bunny"
test_day="240512"
depth="8"
width="256"
epoch="20"

datasets=("knights" "bunny" "dragon") 

python  scripts/llff2neulf.py "${stanford_path}/${stanford2}" --images images --downscale 1 # if you prefer to use the low-resolution images
python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "${test_day}_${stanford2}_uvxy_wire_encX_test" --depth $depth --width $width --LF_mode uvxy --whole_epoch 300 
#python  main_wire_neulf.py "${bmw_path}/${bmw1}" --workspace "${test_day}_${bmw1}_uvxy_wire_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch 300 --render_only

#python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "${test_day}_${stanford1}_vec_wire_encX_test" --depth $depth --width $width --LF_mode vec --whole_epoch 300 


python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_relu_encX" --depth $depth --width $width --LF_mode uvxy --neulf --whole_epoch $epoch 
python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_relu_encX" --depth $depth --width $width --LF_mode vec --neulf --whole_epoch $epoch 
python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_wire_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch 
python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch $epoch 
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_freq_with_relu" --depth $depth --width $width --LF_mode uvxy --act relu --freq_neulf --whole_epoch $epoch 
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --act relu --freq_neulf --whole_epoch $epoch 
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_freq_with_wire" --depth $depth --width $width --LF_mode uvxy --act wire --freq_neulf --whole_epoch $epoch 
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_freq_with_wire" --depth $depth --width $width --LF_mode vec --act wire --freq_neulf --whole_epoch $epoch

python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_relu_encX" --depth $depth --width $width --LF_mode uvxy --neulf --whole_epoch $epoch --test
python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_relu_encX" --depth $depth --width $width --LF_mode vec --neulf --whole_epoch $epoch --test
python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_wire_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch --test
python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch $epoch --test
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_freq_with_relu" --depth $depth --width $width --LF_mode uvxy --act relu --freq_neulf --whole_epoch $epoch --test 
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --act relu --freq_neulf --whole_epoch $epoch --test
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_freq_with_wire" --depth $depth --width $width --LF_mode uvxy --act wire --freq_neulf --whole_epoch $epoch --test
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_freq_with_wire" --depth $depth --width $width --LF_mode vec --act wire --freq_neulf --whole_epoch $epoch --test


python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_relu_encX" --depth $depth --width $width --LF_mode uvxy --neulf --whole_epoch $epoch --render_only
python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_relu_encX" --depth $depth --width $width --LF_mode vec --neulf --whole_epoch $epoch --render_only
python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_wire_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch --render_only
python  main_wire_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch $epoch --render_only
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_freq_with_relu" --depth $depth --width $width --LF_mode uvxy --act relu --freq_neulf --whole_epoch $epoch --render_only
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --act relu --freq_neulf --whole_epoch $epoch --render_only
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_uvxy_freq_with_wire" --depth $depth --width $width --LF_mode uvxy --act wire --freq_neulf --whole_epoch $epoch --render_only
python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "result${test_dat}/${test_day}_${stanford1}_vec_freq_with_wire" --depth $depth --width $width --LF_mode vec --act wire --freq_neulf --whole_epoch $epoch --render_onlys

#python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "${test_day}_${stanford1}_vec_freq_and_sh_with_wire" --depth $depth --width $width --LF_mode vec --act wire --whole_epoch $epoch 


#python  -m pdb main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "${test_day}_${stanford1}_uvxy_freq_with_wire" --depth $depth --width $width --LF_mode uvxy --act wire --whole_epoch 20 --freq_neulf
#python  main_featuresplit_neulf.py "${stanford_path}/${stanford1}" --workspace "${test_day}_${stanford1}_uvxy_freq_with_wire" --depth $depth --width $width --LF_mode uvxy --act wire --whole_epoch 20 --freq_neulf --render_only
#python scripts/llff2neulf.py "${llff_path}/${llff1}" --images images_4 --downscale 4 # if you prefer to use the low-resolution images

#python main_wire_neulf.py  "${stanford_path}/${stanford1}" --workspace "${test_day}_${stanford1}" --depth $depth --width $width --exp_sc --jw_test full
# python  main_wire_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec" --depth $depth --width $width --LF_mode vec --whole_epoch 20 
# python  main_wire_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec" --depth $depth --width $width --LF_mode vec --whole_epoch 20 --render_only

#python main_wire_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec" --depth $depth --width $width --LF_mode vec --whole_epoch 20
#python main_wire_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_uvxy" --depth $depth --width $width --LF_mode uvxy --whole_epoch 20
# python main_wire_neulf.py  $file_name --workspace "${result_name}in" --depth $depth --width $width --exp_sc --jw_test in
# python main_wire_neulf.py  $file_name --workspace "${result_name}out" --depth $depth --width $width --exp_sc --jw_test out


