stanford_path="/data/hmjung/data_backup/NeuLF/dataset/stanford_half"
llff_path="/data/hmjung/data_backup/gaussian-splatting/dataset/nerf_llff_data"

llff1="fern"
stanford1="knights"
test_day="240512"
depth="8"
width="256"

epoch="300"

#python main_wire_neulf.py  "${stanford_path}/${stanford1}" --workspace "${test_day}_${stanford1}" --depth $depth --width $width --exp_sc --jw_test full
#python  main_wire_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec" --depth $depth --width $width --LF_mode vec --whole_epoch 300 
#python  main_wire_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec" --depth $depth --width $width --LF_mode vec --whole_epoch 300  --render_only
# python  main_featuresplit_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --whole_epoch 10  --act relu --freq_neulf
# python  main_featuresplit_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --whole_epoch 10  --act relu --freq_neulf --render_only
# python  main_featuresplit_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec_freq_with_wire" --depth $depth --width $width --LF_mode vec --whole_epoch 10  --act wire --freq_neulf
# python  main_featuresplit_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec_freq_with_wire" --depth $depth --width $width --LF_mode vec --whole_epoch 10  --act wire --freq_neulf --render_only
# python  main_featuresplit_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec_freq_and_wire_with_wire" --depth $depth --width $width --LF_mode vec --whole_epoch 300  --act wire 

#python main_wire_neulf.py  "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec" --depth $depth --width $width --LF_mode vec --whole_epoch 20

python  main_wire_neulf.py "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec_relu_encX" --depth $depth --width $width --LF_mode vec --neulf --whole_epoch $epoch --render_only
python  main_wire_neulf.py "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch $epoch --render_only

python  main_featuresplit_neulf.py "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_vec_freq_with_wire" --depth $depth --width $width --LF_mode vec --act wire --freq_neulf --whole_epoch $epoch --render_only 
python  main_featuresplit_neulf.py "${llff_path}/${llff1}" --workspace "${test_day}_${llff1}_${stanford1}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --act relu --freq_neulf --whole_epoch $epoch --render_only


# python main_wire_neulf.py  $file_name --workspace "${result_name}in" --depth $depth --width $width --exp_sc --jw_test in
# python main_wire_neulf.py  $file_name --workspace "${result_name}out" --depth $depth --width $width --exp_sc --jw_test out


