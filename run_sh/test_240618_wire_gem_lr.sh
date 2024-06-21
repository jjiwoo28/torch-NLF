stanford_path="/data/hmjung/data_backup/NeuLF/dataset/stanford_half"
llff_path="/data/hmjung/data_backup/gaussian-splatting/dataset/nerf_llff_data"
bmw_path="data"
bmw1="bike_1by1_direction1_new"


llff1="fern"
stanford1="knights"
stanford2="bunny"
test_day="240618_wire_knight_lr"
depth="8"
width="256"
epoch="1"
#datasets=("gem" "knights" "bunny" "beans" "flowers" "chess" "bracelet" "bulldozer" "treasure" "truck" "tarot_small" "tarot")  
#datasets=("knights" "gem" "bunny" "bracelet")  
#datasets=("gem" "beans")  
datasets=("knights")  



for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    #python scripts/stanford2neulf.py "${stanford_path}/${dataset}" --images images --downscale 1
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_5e-3" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 5e-3 --no_skips
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_5e-4" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 5e-4 --sigma 40 --omega 49
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_4e-4" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 4e-4 --sigma 40 --omega 40
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_3e-4" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 3e-4 --sigma 40 --omega 40
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_2e-4" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 2e-4 --sigma 40 --omega 40
    python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_1e-4" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 1e-4 --sigma 40 --omega 40
    python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_5e-5" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 5e-5 --sigma 40 --omega 40
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${test_day}/${test_day}_${dataset}_uvxy_wire_encX_5e-6" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 1 --loss_coeff 1000 --lr 5e-6
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