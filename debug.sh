stanford_path="/data/hmjung/data_backup/NeuLF_rgb/dataset/stanford_half"
llff_path="/data/hmjung/data_backup/gaussian-splatting/dataset/nerf_llff_data"
bmw_path="data"
bmw1="bike_1by1_direction1_new"


llff1="fern"
stanford1="knights"
stanford2="bunny"
test_day="240514"
depth="8"
width="256"
epoch="20"

datasets=( "treasure" "gem" "knights" "bunny" "beans" "flowers" "chess" "bracelet" "bulldozer" "treasure" "truck" "tarot_small" "tarot")  


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    python -m pdb scripts/llff2neulf.py "${stanford_path}/${dataset}" --images images --downscale 1

    # python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch 10
    # python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch 10 -test
    

    python -m pdb main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "result_debug${test_day}/${test_day}_${dataset}_uvxy_relu_encX_debug" --depth $depth --width $width --LF_mode uvxy --neulf --whole_epoch $epoch 
    python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_relu_encX" --depth $depth --width $width --LF_mode uvxy --neulf --whole_epoch $epoch --render_only
    python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_relu_encX" --depth $depth --width $width --LF_mode vec --neulf --whole_epoch $epoch 
    python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_relu_encX" --depth $depth --width $width --LF_mode vec --neulf --whole_epoch $epoch --render_only

    # python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_wire_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch 
    # python main_wire_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_wire_encX" --depth $depth --width $width --LF_mode vec --whole_epoch $epoch 

    # python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_freq_with_relu" --depth $depth --width $width --LF_mode uvxy --act relu --freq_neulf --whole_epoch $epoch 
    # python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_vec_freq_with_relu" --depth $depth --width $width --LF_mode vec --act relu --freq_neulf --whole_epoch $epoch 
    # python main_featuresplit_neulf.py "${stanford_path}/${dataset}" --workspace "/data/hmjung/result${test_day}/${test_day}_${dataset}_uvxy_freq_with_wire" --depth $depth --width $width --LF_mode uvxy --act wire --freq_neulf --whole_epoch $epoch 
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