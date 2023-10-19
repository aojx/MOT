from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd

# raw data 파일
data = pd.read_csv("sample.csv")

# NaN 값을 포함하는 행을 제거
data.dropna(subset=['appln_title', 'appln_abstract'], inplace=True)

# 학습 데이터와 테스트 데이터에서 제목과 초록을 추출
# 각 row 별로  appln_title 컬럼과 appln_abstract 컬럼의 데이터를 합쳐 리스트 만들기
documents = data['appln_title'] + ' ' + data['appln_abstract']

# 사용자 정의 불용어 목록(콤마로 구분하여 계속 추가 가능)
custom_stop_words = [
        'method','methods','apparatus','using','en','invention',
        'second','comprises','comprising','material','having','present','provided','relates',
        'process','device','includes','arranged','data','end','body','box',
        'collecting','collection','connected','discloses','field'
]

# 불용어 목록 결합
all_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words

# TF-IDF 벡터화
# 조금 더 많은 단어를 추출해보고 싶다면 max_features 를 늘리기
vectorizer = TfidfVectorizer(max_features=20,stop_words=all_stop_words,ngram_range=(1,3),min_df=10,token_pattern=r'\b[a-zA-Z_][a-zA-Z_]+\b')
tfidf_matrix = vectorizer.fit_transform(documents)

# 단어 목록
terms = vectorizer.get_feature_names_out()

# TF-IDF 값 및 단어 출력
tfidf_values = tfidf_matrix.toarray()[0]
df = pd.DataFrame({'Term': terms, 'TF-IDF': tfidf_values})
#df = df.sort_values(by=['TF-IDF'], ascending=False)

# 출력할 목록의 갯수를 제한하지 않음
pd.set_option("display.max_rows", None)

print(df)
