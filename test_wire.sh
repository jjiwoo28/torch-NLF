file_name="data/bike_1by1_direction1_new"
result_name="240505_bike_1by1_direction1_"
depth="8"
width="256"

python main_wire_neulf.py  $file_name --workspace "${result_name}full" --depth $depth --width $width --exp_sc --jw_test full
python main_wire_neulf.py  $file_name --workspace "${result_name}in" --depth $depth --width $width --exp_sc --jw_test in
python main_wire_neulf.py  $file_name --workspace "${result_name}out" --depth $depth --width $width --exp_sc --jw_test out


