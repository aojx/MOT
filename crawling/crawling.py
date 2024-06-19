import requests
from bs4 import BeautifulSoup

url = "http://www.investing.com/commodities/metals"

# requests를 사용하여 웹페이지 가져오기
response = requests.get(url)
print(f"status_code:{response.status_code}\n--------------------")

if response.status_code == 200:

    soup = BeautifulSoup(response.text, 'html.parser')
    table_selector = "table#cross_rate_1"

    # 테이블에서 데이터를 찾기
    table = soup.select_one(table_selector)

    if table:
        # ID가 "pair-일련번호"인 각 행을 찾기
        for row in table.select_one('tbody').select("tr[id^='pair_']"):
            # 각 행에서 "pid-일련번호-last" 클래스를 가진 열 찾기
            pid_number = row['id'].replace('pair_', '')  # 일련번호 추출
            column_selector = f".pid-{pid_number}-last"
            column_data = row.select_one(column_selector)
            name = row.select_one(".elp").select_one("a").text

            if column_data:
                # 데이터 추출
                price = column_data.text.strip()
                print(f"{name}({column_selector}): {price}")
            else:
                print(f"pid-{pid_number}-last 클래스를 찾을 수 없습니다.")
    else:
        print("테이블을 찾을 수 없습니다.")
else:
    print(f"Error: {response.status_code}")
