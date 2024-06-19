from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
from multiprocessing import Pool

# 데이터 로딩
data = pd.read_csv("patents_2021.csv")
#data = pd.read_csv("patents_g06q20_all_with_title_abstract.csv")

data.dropna(subset=['appln_title', 'appln_abstract', 'APPLN_FILING_YEAR'], inplace=True)

# 고유한 연도 목록 추출
unique_years = sorted(data['APPLN_FILING_YEAR'].unique())

# 사용자 정의 불용어 목록
custom_stop_words = [
    # 불용어 목록 이곳에 작성
    'method','methods','apparatus','using','en','invention',
    'second','comprises','comprising','material','having','present','provided','relates',
    'includes','arranged','end','body','box',
    'collecting','collection','discloses','field',
    'la','el','et','une','means','based','associated','according','corresponding',
    'receiving','des','said','selected','le','dad','una','que',
    'including','fig','unit','code','number','read','stored',
    'target','time','control','value','line','signals','key','token','use','pour',
    'head','quantity','dispensing','light','shown','used','keys','host',
    'area','program','form','par','du',
    'systems','plurality','providing','vehicle','received','configured','transactions',
    'donnees','est','financial','determining','predetermined'

    #'device','terminal','card','identification','authentication','communication','management','transaction','account','payment',
    #'electronic','application','server','data','module','information','service','customer','register','operation','process','user','processing'
    #'store','access','request','connected'
]

# 불용어 목록 결합
all_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words

# TF-IDF 계산을 위한 함수
def calculate_tfidf_for_year(year_data):
    year, data_chunk = year_data
    print(f"Processing year: {year}")
    documents = data_chunk['appln_title'] + ' ' + data_chunk['appln_abstract']
    tfidf_vectorizer = TfidfVectorizer(
        max_features=20,
	ngram_range=(1, 2),
        stop_words=all_stop_words,
        min_df=0.02,
        token_pattern=r'\b[a-zA-Z_][a-zA-Z1-9\-_]+\b',
        max_df=0.95
    )
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        terms = tfidf_vectorizer.get_feature_names_out()
        avg_tfidf_values = tfidf_matrix.toarray().mean(axis=0)

        # 각 특허별 TF-IDF 값 계산
        individual_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)
        individual_tfidf.insert(0, 'Patent_ID', data_chunk['APPLN_ID'].values)
        
        # 피처별 TF-IDF 값 출력
        feature_avg_tfidf = individual_tfidf.drop(columns=['Patent_ID']).mean(axis=0).reset_index()
        feature_avg_tfidf.columns = ['Term', 'Avg_TF-IDF']
        feature_avg_tfidf = feature_avg_tfidf.sort_values(by='Avg_TF-IDF', ascending=False)

        print(f"Feature average TF-IDF values for year {year}:\n", feature_avg_tfidf)

        # 연도별 평균 TF-IDF 값 데이터프레임 생성
        avg_tfidf_df = pd.DataFrame({'Year': [year] * len(terms), 'Term': terms, 'TF-IDF': avg_tfidf_values})
        
        return avg_tfidf_df, individual_tfidf
    except ValueError as e:
        print(f"Skipping year {year}: {e}")
        return pd.DataFrame({'Year': [], 'Term': [], 'TF-IDF': []}), pd.DataFrame()

# 메인 함수
def main():
    year_data_pairs = [(year, data[data['APPLN_FILING_YEAR'] == year]) for year in unique_years]

    with Pool(processes=4) as pool:  # 사용 가능한 CPU 코어 수에 맞게 설정
        results = pool.map(calculate_tfidf_for_year, year_data_pairs)

    avg_tfidf_results = [result[0] for result in results]
    individual_tfidf_results = [result[1] for result in results if not result[1].empty]

    avg_tfidf_df = pd.concat(avg_tfidf_results, ignore_index=True)
    individual_tfidf_df = pd.concat(individual_tfidf_results, ignore_index=True)

    avg_tfidf_df = avg_tfidf_df.sort_values(by=['Year', 'TF-IDF'], ascending=[True, False])
    pd.set_option("display.max_rows", None)

    # CSV 파일로 저장
    avg_tfidf_df.to_csv("average_tfidf_by_year.csv", index=False)
    individual_tfidf_df.to_csv("individual_patent_tfidf.csv", index=False)

    print("Average TF-IDF Values by Year:")
    print(avg_tfidf_df)

    print("\nIndividual Patent TF-IDF Values:")
    print(individual_tfidf_df)

if __name__ == "__main__":
    main()
