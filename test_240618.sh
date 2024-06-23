stanford_path="/data/stanford_half"
llff_path="/data/nerf_llff_data"
bmw_path="/data/bmw_dataset"


test_day="240621_torch_neulf_stanford"
depth="8"
width="256"
epoch="1000"
#datasets=("gem" "knights" "bunny" "beans" "flowers" "chess" "bracelet" "bulldozer" "treasure" "truck" "tarot_small" "tarot")  
datasets=( "gem" "bunny" "bracelet"  "flowers" "tarot" "chess" "bulldozer" "treasure")  
#datasets=("gem")  



for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    #python scripts/stanford2neulf.py "${stanford_path}/${dataset}" --images images --downscale 1
    python main_neulf.py "${stanford_path}/${dataset}" --workspace "/data/result/result${test_day}/${test_day}_${dataset}_uvxy_relu_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch --eval_interval 10 
    #python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "${test_day}_${dataset}_vec_relu_encX" --depth $depth --width $width --LF_mode vec --neulf --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000
    # python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_wire_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch --render_only
    # python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch $epoch --render_only
    #python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "${test_day}_${dataset}_uvxy_freq_with_relu" --depth $depth --width $width --LF_mode uvxy --act relu --freq_neulf --whole_epoch $epoch --eval_interval 1
    #python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "${test_day}_${dataset}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --act relu --freq_neulf --whole_epoch $epoch --eval_interval 1
    # python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}${test_day}_${dataset}_uvxy_freq_with_wire" --depth $depth --width $width --LF_mode uvxy --act wire --freq_neulf --whole_epoch $epoch 
    # python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_freq_with_wire" --depth $depth --width $width --LF_mode vec --act wire --freq_neulf --whole_epoch $epoch 
done


# for dataset in "${datasets[@]}"; do
#     echo "Processing $dataset"
#     python scripts/llff2neulf.py "${stanford_path}/${dataset}" --images images --downscale 1
#     python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_relu_encX" --depth $depth --width $width --LF_mode uvxy --neulf --whole_epoch $epoch --test
#     python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_relu_encX" --depth $depth --width $width --LF_mode vec --neulf --whole_epoch $epoch --test
#     python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_wire_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch --test
#     python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch $epoch --test
#     python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_freq_with_relu" --depth $depth --width $width --LF_mode uvxy --act relu --freq_neulf --whole_epoch $epoch --test
#     python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --act relu --freq_neulf --whole_epoch $epoch --test
#     python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_freq_with_wire" --depth $depth --width $width --LF_mode uvxy --act wire --freq_neulf --whole_epoch $epoch --test
#     python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_freq_with_wire" --depth $depth --width $width --LF_mode vec --act wire --freq_neulf --whole_epoch $epoch --test
# done


# for dataset in "${datasets[@]}"; do
#     echo "Processing $dataset"
#     python scripts/llff2neulf.py "${stanford_path}/${dataset}" --images images --downscale 1
#     python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_relu_encX" --depth $depth --width $width --LF_mode uvxy --neulf --whole_epoch $epoch --render_only
#     python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_relu_encX" --depth $depth --width $width --LF_mode vec --neulf --whole_epoch $epoch --render_only
#     python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_wire_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch --render_only
#     python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch $epoch --render_only
#     python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_freq_with_relu" --depth $depth --width $width --LF_mode uvxy --act relu --freq_neulf --whole_epoch $epoch --render_only
#     python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --act relu --freq_neulf --whole_epoch $epoch --render_only
#     python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_freq_with_wire" --depth $depth --width $width --LF_mode uvxy --act wire --freq_neulf --whole_epoch $epoch --render_only
#     python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_freq_with_wire" --depth $depth --width $width --LF_mode vec --act wire --freq_neulf --whole_epoch $epoch --render_only
# done
