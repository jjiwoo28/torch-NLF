stanford_path="/data/stanford_half"
llff_path="/data/nerf_llff_data"
bmw_path="/data/bmw_dataset"


test_day="240702_asem_json_debug"
depth="8"
width="256"
epoch="3"

datasets=( "gem" "bunny" "bracelet")  

result_path="/data/result/result${test_day}"


for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    #python scripts/stanford2neulf.py "${stanford_path}/${dataset}" --images images --downscale 1
    python main_neulf.py "${stanford_path}/${dataset}" --workspace "${result_path}/${test_day}_${dataset}_uvxy_relu_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch --eval_interval 1 

done

python asem_json.py  "/data/result/result${test_day}" "result_json/${test_day}"

