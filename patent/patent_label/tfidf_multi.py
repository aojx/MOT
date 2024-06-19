from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
from multiprocessing import Pool

# 데이터 로딩
#data = pd.read_csv("patents_g06q20_all_with_title_abstract.csv")
data = pd.read_csv("patents_2022.csv")
data.dropna(subset=['appln_title', 'appln_abstract', 'APPLN_FILING_YEAR'], inplace=True)

# 고유한 연도 목록 추출
unique_years = sorted(data['APPLN_FILING_YEAR'].unique())

# 사용자 정의 불용어 목록
custom_stop_words = [
    # 불용어 목록 이곳에 작성
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
        'donnees','est','financial','terminal','determining'
]

# 불용어 목록 결합
all_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words

# TF-IDF 계산을 위한 함수
def calculate_tfidf_for_year(year_data):
    year, data_chunk = year_data
    print(f"Processing year: {year}")
    documents = data_chunk['appln_title'] + ' ' + data_chunk['appln_abstract']
    tfidf_vectorizer = TfidfVectorizer(
        max_features=18,
        stop_words=all_stop_words,
        #ngram_range=(1, 2),
        min_df=0.02,  # 낮춘 min_df 값
        token_pattern=r'\b[a-zA-Z_][a-zA-Z1-9\-_]+\b',
        max_df=0.90  # 높인 max_df 값
    )
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        terms = tfidf_vectorizer.get_feature_names_out()
        avg_tfidf_values = tfidf_matrix.toarray().mean(axis=0)
        return pd.DataFrame({'Year': [year] * len(terms), 'Term': terms, 'TF-IDF': avg_tfidf_values})
    except ValueError as e:
        print(f"Skipping year {year}: {e}")
        return pd.DataFrame({'Year': [], 'Term': [], 'TF-IDF': []})

# 메인 함수
def main():
    year_data_pairs = [(year, data[data['APPLN_FILING_YEAR'] == year]) for year in unique_years]
    
    with Pool(processes=4) as pool:  # 사용 가능한 CPU 코어 수에 맞게 설정
        results = pool.map(calculate_tfidf_for_year, year_data_pairs)
    
    result_df = pd.concat(results, ignore_index=True)
    result_df = result_df.sort_values(by=['Year', 'TF-IDF'], ascending=[True, False])
    pd.set_option("display.max_rows", None)
    print(result_df)

if __name__ == "__main__":
    main()

