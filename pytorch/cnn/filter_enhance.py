import os
import cv2
import shutil
import numpy as np
import math
from scipy.ndimage import gaussian_filter1d

def vignetting_correction(image):
    height, width = image.shape[:2]

    # 보통 이미지 크기의 1/6 ~ 1/4 을 σ(sigma) 값으로 설정한다고 함
    # 반복 테스트를 통해 적정 sigma 값을 찾기 (이미지 크기와 동일한 값이 적절하다고 판단하면 그렇게 사용)
    width = 384
    sigma = width / 1
    
    kernel_x = cv2.getGaussianKernel(width, sigma)
    kernel_y = cv2.getGaussianKernel(height, sigma)
    kernel = kernel_y * kernel_x.T

    mask = 255 * kernel / np.linalg.norm(kernel)
    
    output = image.copy()

    for i in range(3):
        output[:,:,i] = output[:,:,i] / mask

    output = np.clip(output, 0, 255)

    #cv2.imshow('Original', image)
    #cv2.imshow('Vignette', output)
    #cv2.waitKey(0)
    return output


def crop_and_resize_images(original_dir, resized_dir, target_size=(256, 256), crop_start=(800, 400), crop_size=2550):
    # resized 디렉토리가 이미 존재하면 삭제
    if os.path.exists(resized_dir):
        shutil.rmtree(resized_dir)
    
    # original 디렉토리 내의 모든 파일 및 디렉토리 목록을 가져옴
    for root, dirs, files in os.walk(original_dir):
        # 해당 디렉토리의 하위 디렉토리를 찾아서, resized 디렉토리에 같은 구조로 생성
        relative_path = os.path.relpath(root, original_dir)
        resized_subdir = os.path.join(resized_dir, relative_path)
        os.makedirs(resized_subdir, exist_ok=True)
        
        # 디렉토리 내의 모든 파일에 대해 작업
        for file in files:
            # 이미지 파일인지 확인
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                # 이미지 로드
                original_path = os.path.join(root, file)
                image = cv2.imread(original_path)

                # 이미지 크롭
                x, y = crop_start
                cropped_image = image[y:y+crop_size, x:x+crop_size]
                
                # 이미지 리사이징
                resized_image = cv2.resize(cropped_image, target_size)
                
                # 리사이징된 이미지 저장
                resized_path = os.path.join(resized_subdir, file)
                cv2.imwrite(resized_path, resized_image)
                print(f"Resized and saved: {resized_path}")

def filter_images(original_dir, filtered_dir):
    # filtered 디렉토리가 이미 존재하면 삭제
    if os.path.exists(filtered_dir):
        shutil.rmtree(filtered_dir)
    
    # original 디렉토리 내의 모든 파일 및 디렉토리 목록을 가져옴
    for root, dirs, files in os.walk(original_dir):
        # 해당 디렉토리의 하위 디렉토리를 찾아서, resized 디렉토리에 같은 구조로 생성
        relative_path = os.path.relpath(root, original_dir)
        filtered_subdir = os.path.join(filtered_dir, relative_path)
        os.makedirs(filtered_subdir, exist_ok=True)
        
        # 디렉토리 내의 모든 파일에 대해 작업
        for file in files:
            # 이미지 파일인지 확인
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                # 이미지 로드
                original_path = os.path.join(root, file)
                image = cv2.imread(original_path)

                image = vignetting_correction(image)
                image = sobel_filter(image)

                #image = saturate_contrast2(image, 1.2)  # 명암비 조정

                # 필터적용한 이미지 저장
                filtered_path = os.path.join(filtered_subdir, file)
                cv2.imwrite(filtered_path, image)
                print(f"Filtered and saved: {filtered_path}")

def saturate_contrast2(p, num):
    pic = p.copy()
    pic = pic.astype('int32')
    pic = np.clip(pic+(pic-128)*num, 0, 255)
    pic = pic.astype('uint8')
    return pic

def sobel_filter(img):
    # 소벨 API를 생성해서 엣지 검출
    sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, -1, 0, 1, ksize=3) 

    return sobelx+sobely

# 원본 이미지가 있는 디렉토리 경로
original_directory = "original"
# 리사이징된 이미지를 저장할 디렉토리 경로
resized_directory = "resized"
# 리사이징된 이미지를 저장할 디렉토리 경로
filtered_directory = "filtered"

# 이미지 크롭 및 리사이징 실행
crop_and_resize_images(original_directory, resized_directory, target_size=(384, 384), crop_start=(810, 400), crop_size=2550)

# 필터 실행
filter_images(resized_directory, filtered_directory)


