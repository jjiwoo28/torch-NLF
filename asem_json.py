import os
import shutil
import argparse

def copy_json_files(source_folder, target_folder):
    # 대상 폴더 생성 (존재하지 않는 경우)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
        print(f"Created target folder: {target_folder}")

    # source_folder를 순회하며 .json 파일 찾기
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.json'):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_folder, file)

                # 파일을 대상 폴더로 복사
                shutil.copy(source_path, target_path)
                print(f"Copied: {source_path} to {target_path}")

def main():
    parser = argparse.ArgumentParser(description="Copy JSON files from source to target folder")
    parser.add_argument("source_folder", type=str, help="Source folder path")
    parser.add_argument("target_folder", type=str, help="Target folder path")

    args = parser.parse_args()

    copy_json_files(args.source_folder, args.target_folder)

if __name__ == "__main__":
    main()
