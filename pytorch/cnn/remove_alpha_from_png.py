import os
from PIL import Image

def remove_alpha_channel(input_dir, output_dir):
    # 출력 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 입력 디렉토리 내의 모든 파일을 순회
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # 알파 채널 제거
            img = img.convert("RGB")

            # 변환된 이미지를 출력 디렉토리에 저장
            output_path = os.path.join(output_dir, filename)
            img.save(output_path)
            print(f"Saved {output_path}")

# 사용 예시
input_directory = "./filtered/NG"  # 원본 이미지가 있는 디렉토리
output_directory = "./filtered/NG2"  # 알파 채널이 제거된 이미지를 저장할 디렉토리

remove_alpha_channel(input_directory, output_directory)