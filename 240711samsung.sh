stanford_path="/data/stanford_half"
llff_path="/data/nerf_llff_data"
bmw_path="/data/bmw_dataset"


test_day="240711_samsung_reult"
depth="8"
width="256"
epoch="1000"

datasets=("knights3")  

result_path="/data/result/result${test_day}"


for dataset in "${datasets[@]}"; do

    echo "Processing $dataset"
    python scripts/stanford2neulf_resize.py "${stanford_path}/${dataset}" --images images --downscale 1 --grid 3
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${result_path}/${test_day}_${dataset}_uvxy_wire_encX_5e-4" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 10 --loss_coeff 1000 --lr 5e-4 
    python main_neulf.py "${stanford_path}/${dataset}" --workspace "${result_path}/${test_day}_${dataset}_uvxy_relu_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch --eval_interval 10 

done

datasets=("knights5") 

for dataset in "${datasets[@]}"; do

    echo "Processing $dataset"
    python scripts/stanford2neulf_resize.py "${stanford_path}/${dataset}" --images images --downscale 1 --grid 5
    #python main_wire_neulf_wire.py "${stanford_path}/${dataset}" --workspace "${result_path}/${test_day}_${dataset}_uvxy_wire_encX_5e-4" --depth $depth --width $width --LF_mode uvxy  --whole_epoch $epoch --eval_interval 10 --loss_coeff 1000 --lr 5e-4 
    python main_neulf.py "${stanford_path}/${dataset}" --workspace "${result_path}/${test_day}_${dataset}_uvxy_relu_encX" --depth $depth --width $width --LF_mode uvxy --whole_epoch $epoch --eval_interval 10 

done


python asem_json.py  "/data/result/result${test_day}" "result_json/${test_day}"

