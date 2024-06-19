######################################################################
#
# - input.csv 를 읽어서 output.csv 를 생성하는 python 프로그램입니다.
# - input.csv 형식
#   특허출원번호, 인용특허목록(구분자'|'), 피인용특허목록(구분자'|')
#
######################################################################

import csv

# 입력 파일 이름과 출력 파일 이름을 지정합니다.
input_filename = 'input.csv'
output_filename = 'output.csv'

# 입력 파일을 열고 출력 파일을 생성합니다.
with open(input_filename, 'r', encoding='utf-8-sig') as input_file, open(output_filename, 'w', newline='', encoding='utf-8-sig') as output_file:
    # CSV 파일 리더와 라이터 객체를 생성합니다.
    csv_reader = csv.reader(input_file)
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(['application','citation','type'])

    # 각 행을 처리합니다.
    for row in csv_reader:
        patent_number = row[0].strip()  # 특허 번호 trim 처리
        citation_numbers = [num.strip() for num in row[1].split('|')]  # 인용 특허 번호 trim 처리
        cited_by_numbers = [num.strip() for num in row[2].split('|')]  # 피인용 특허 번호 trim 처리

        # 인용 특허 번호가 없는 경우 "인용 없음"으로 처리
        if citation_numbers == ['']:
            csv_writer.writerow([patent_number, '', 'NO_FWD'])
        else:
            # 인용 특허 번호를 처리하고 출력 파일에 기록합니다.
            for citation_number in citation_numbers:
                csv_writer.writerow([patent_number, citation_number, 'FWD'])

        # 피인용 특허 번호가 없는 경우 "피인용 없음"으로 처리
        if cited_by_numbers == ['']:
            csv_writer.writerow([patent_number, '', 'NO_BWD'])
        else:
            # 피인용 특허 번호를 처리하고 출력 파일에 기록합니다.
            for cited_by_number in cited_by_numbers:
                csv_writer.writerow([patent_number, cited_by_number, 'BWD'])
