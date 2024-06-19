from difflib import SequenceMatcher
import pandas as pd
from multiprocessing import Pool, cpu_count

stop_words = [ "company", "corporation", "international", "technologies", "electronics", "technology" ]

# 특정 단어 그룹
target_groups = [["alibaba", "alipay"], ["visa"], ["samsung"], ["sony"], ["mastercard"], ["lg"], ["amazon"], ["shinhan"],["sk"]]

def get_group_name(company_name):
    # 각 그룹에 속하는 단어 찾기
    for group in target_groups:
        for word in group:
            if word in company_name:
                return tuple(group)
    return tuple([company_name])

def similarity(a, b):
    # 불용어를 제거하고 소문자로 변환한 후 유사성 계산
    a = a.lower()
    b = b.lower()
    a_words = [word for word in a.split() if word not in stop_words]
    b_words = [word for word in b.split() if word not in stop_words]
    
    # 기업명을 해당 그룹으로 분류
    a_group = get_group_name(a)
    b_group = get_group_name(b)

    # 동일한 그룹에 속한 경우만 그룹으로 묶기
    if a_group == b_group:
        return 1.0  # 유사성 점수를 1로 설정하여 그룹핑
    else:
        return SequenceMatcher(None, ' '.join(a_words), ' '.join(b_words)).ratio()

def process_group(args):
    name, groups, threshold = args
    found = False
    for group in groups:
        for existing_name in group:
            if similarity(name, existing_name) >= threshold:
                group.append(name)
                found = True
                break
        if found:
            break
    if not found:
        groups.append([name])
    return groups

if __name__ == '__main__':
   # CSV 파일로부터 데이터 읽어오기
   data = pd.read_csv("patents_g06q20_all_company.csv")  # CSV 파일 경로를 지정하세요
   company_names = data['psn_name'].tolist()

   # 그룹화 실행
   threshold = 0.85 # 유사성 임계값 설정
   groups = []

   # 멀티프로세싱을 위한 풀 생성
   with Pool(processes=cpu_count()) as pool:
      args_list = [(name, groups, threshold) for name in company_names]
      # 각 기업명에 대해 비동기적으로 그룹화 작업 수행
      results = pool.map(process_group, args_list)

   #print(results)

   # 결과를 파일에 출력
   output_file_path = "company_group_result.txt"
   with open(output_file_path, "w") as output_file:
      for i, group in enumerate(results):
          output_file.write(f"그룹 {i + 1}: {', '.join(map(str, group))}\n")

   print(f"결과가 {output_file_path} 파일에 저장되었습니다.")

   # 결과 출력
   #for i, group in enumerate(groups):
   #    print(f"그룹 {i + 1}: {', '.join(group)}")
