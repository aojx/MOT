from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd

# raw data 파일
data = pd.read_csv("patents_g06q20_all_with_title_abstract.csv")

# NaN 값을 포함하는 행을 제거
data.dropna(subset=['appln_title', 'appln_abstract', 'APPLN_FILING_YEAR'], inplace=True)


# 연도별로 그룹화하여 TF-IDF 계산
result_frames = []


# 고유한 연도 목록 추출
unique_years = sorted(data['APPLN_FILING_YEAR'].unique())

# 사용자 정의 불용어 목록(콤마로 구분하여 계속 추가 가능)
custom_stop_words = [
 	'method','methods','apparatus','using','en','invention',
 	'second','comprises','comprising','material','having','present','provided','relates',
 	'process','device','includes','arranged','data','end','body','box',
 	'collecting','collection','connected','discloses','field',
 	'la','el','et','une','means','based','associated','according','corresponding',
 	'receiving','des','said','selected','user','le','dad','una','que',
 	'module','information','including','management','transaction','account',
 	'processing','request','payment','fig','service','server',
 	'card','unit','code','number','read','stored','communication',
 	'target','time','control','customer','electronic','identification',
 	'application','authentication','value','line','signals','key','token',
 	'use','pour','head','quantity','store','dispensing','light','shown','used','keys','host',
 	'register','operation','area','program','access','form','par','du',
 	'systems','plurality','providing','vehicle','received','configured','transactions',
 	'donnees','est','financial','terminal'

]

# 불용어 목록 결합
all_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words

for year in unique_years:
    if year < 2000 or year > 2022:
        continue
    print(year)
    # 해당 연도의 데이터만 추출
    year_data = data[data['APPLN_FILING_YEAR'] == year]

    # 학습 데이터와 테스트 데이터에서 제목과 초록을 추출
    # 각 row 별로  appln_title 컬럼과 appln_abstract 컬럼의 데이터를 합쳐 리스트 만들기
    documents = year_data['appln_title'] + ' ' + year_data['appln_abstract']
    #documents = year_data['appln_abstract']
    #documents = year_data['appln_title']

    # TF-IDF 벡터화
    # 조금 더 많은 단어를 추출해보고 싶다면 max_features 를 늘리기
    tfidf_vectorizer = TfidfVectorizer(max_features=10,stop_words=all_stop_words,ngram_range=(1,2),min_df=0.02,token_pattern=r'\b[a-zA-Z_][a-zA-Z1-9\-_]+\b',max_df=0.85)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    print(tfidf_vectorizer)
    # 단어 목록
    terms = tfidf_vectorizer.get_feature_names_out()
    print(terms)

    # TF-IDF 값 및 단어 출력
    #tfidf_values = tfidf_matrix.toarray()[0]
    print(tfidf_matrix.toarray())
    # 각 단어의 평균 TF-IDF 값을 계산
    avg_tfidf_values = tfidf_matrix.toarray().mean(axis=0)

    df = pd.DataFrame({'Year': [year] * len(terms), 'Term': terms, 'TF-IDF': avg_tfidf_values})
    result_frames.append(df)

# 결과를 하나의 데이터프레임으로 결합
result_df = pd.concat(result_frames, ignore_index=True)

# 연도별로 정렬
result_df = result_df.sort_values(by=['Year', 'TF-IDF'], ascending=[True, True])

# 출력할 목록의 갯수를 제한하지 않음
pd.set_option("display.max_rows", None)

print(result_df)
