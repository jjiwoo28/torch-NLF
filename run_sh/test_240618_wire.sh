stanford_path="/data/stanford_half"
llff_path="/data/nerf_llff_data"
bmw_path="/data/bmw_dataset"


test_day="240621_torch_NLF_wire_test"
depth="8"
width="256"
epoch="1000"
#datasets=("gem" "knights" "bunny" "beans" "flowers" "chess" "bracelet" "bulldozer" "treasure" "truck" "tarot_small" "tarot")  
datasets=("bunny" "bracelet"  "flowers" "tarot" "chess" "bulldozer" "treasure")  





for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    #python scripts/stanford2neulf.py "${stanford_path}/${dataset}" --images images --downscale 1
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_5e-3" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 5e-3 --no_skips
    python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "/data/result/result${test_day}/${test_day}_${dataset}_uvxy_wire_encX_5e-4" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 10 --loss_coeff 1000 --lr 5e-4 

    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_5e-6" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 5e-6
    #python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "${test_day}_${dataset}_vec_relu_encX" --depth $depth --width $width --LF_mode vec --neulf --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000
    # python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_wire_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch --render_only
    # python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch $epoch --render_only
    #python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "${test_day}_${dataset}_uvxy_freq_with_relu" --depth $depth --width $width --LF_mode uvxy --act relu --freq_neulf --whole_epoch $epoch --eval_interval 1
    #python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "${test_day}_${dataset}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --act relu --freq_neulf --whole_epoch $epoch --eval_interval 1
    # python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}${test_day}_${dataset}_uvxy_freq_with_wire" --depth $depth --width $width --LF_mode uvxy --act wire --freq_neulf --whole_epoch $epoch 
    # python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_freq_with_wire" --depth $depth --width $width --LF_mode vec --act wire --freq_neulf --whole_epoch $epoch 
done

python asem_jdon.py "/data/result/result${test_day}" "result_json/result${test_day}"

datasets=("gem"  "bracelet" "chess") 

for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    #python scripts/stanford2neulf.py "${stanford_path}/${dataset}" --images images --downscale 1
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_5e-3" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 5e-3 --no_skips
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_5e-4" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 5e-4 
    python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "/data/result/result${test_day}/${test_day}_${dataset}_uvxy_wire_encX_5e-5" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 10 --loss_coeff 1000 --lr 5e-5 
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_5e-6" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 5e-6
    #python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "${test_day}_${dataset}_vec_relu_encX" --depth $depth --width $width --LF_mode vec --neulf --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000
    # python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_wire_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch --render_only
    # python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch $epoch --render_only
    #python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "${test_day}_${dataset}_uvxy_freq_with_relu" --depth $depth --width $width --LF_mode uvxy --act relu --freq_neulf --whole_epoch $epoch --eval_interval 1
    #python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "${test_day}_${dataset}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --act relu --freq_neulf --whole_epoch $epoch --eval_interval 1
    # python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}${test_day}_${dataset}_uvxy_freq_with_wire" --depth $depth --width $width --LF_mode uvxy --act wire --freq_neulf --whole_epoch $epoch 
    # python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_freq_with_wire" --depth $depth --width $width --LF_mode vec --act wire --freq_neulf --whole_epoch $epoch 
done

python asem_jdon.py "/data/result/result${test_day}" "result_json/result${test_day}"

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
