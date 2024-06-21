import os
import shutil

def copy_json_files(source_folder, target_folder):
    # 대상 폴더 생성 (존재하지 않는 경우)
    os.makedirs(target_folder, exist_ok=True)

    # source_folder를 순회하며 .json 파일 찾기
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.json'):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_folder, file)

                # 파일을 대상 폴더로 복사
                shutil.copy(source_path, target_path)
                print(f"Copied: {source_path} to {target_path}")

# 사용 예시
source_folder1 = "/data/hmjung/result240618_torch_neulf_stanford_skips_test240618" 
#source_folder2 = "/data/hmjung/result240518_test_2" 
target_folder =  "result_json_240618_torch_neulf_stanford_skips_test240618"
copy_json_files(source_folder1, target_folder)
#copy_json_files(source_folder2, target_folder)
