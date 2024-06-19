import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 학습 데이터와 테스트 데이터 파일을 각각 로드
train_data = pd.read_csv("greenhouse_gas.csv")  # 학습 데이터 파일
test_data = pd.read_csv("test_data.csv")  # 테스트 데이터 파일

# NaN 값을 포함하는 행을 제거
train_data.dropna(subset=['appln_title', 'appln_abstract'], inplace=True)
test_data.dropna(subset=['subject', 'abstract'], inplace=True)

# 학습 데이터와 테스트 데이터에서 제목과 초록을 추출
#X_train = train_data['subject'] + ' ' + train_data['abstract']
X_train = train_data['appln_title'] + ' ' + train_data['appln_abstract']
y_train = train_data['nat_phase']
X_test = test_data['subject'] + ' ' + test_data['abstract']
y_test = test_data['type']

# 모델 선택 및 학습
vectorizer = TfidfVectorizer(max_features=200,stop_words='english',ngram_range=(1,3),min_df=5,token_pattern=r'\b[a-zA-Z_][a-zA-Z_]+\b')
X_train_tfidf = vectorizer.fit_transform(X_train)
clf = RandomForestClassifier(n_estimators=500, ## 붓스트랩 샘플 개수 또는 base_estimator 개수
            criterion='entropy', ## 불순도 측도
            max_depth=100, ## 개별 나무의 최대 깊이
            max_features='sqrt', ## 매 분리시 랜덤으로 뽑을 변수 개수
            max_samples=1.0, ## 붓스트랩 샘플 비율 => 1이면 학습데이터를 모두 샘플링한다.
            bootstrap=True, ## 복원 추출,  False이면 비복원 추출
            oob_score=True, ## Out-of-bag 데이터를 이용한 성능 계산
            random_state=50)  # 100개의 결정 트리를 사용
clf.fit(X_train_tfidf, y_train)

ftr_importances_values = clf.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Top 20 Feature Importances')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()


# 모델 저장
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(clf, 'random_forest_model.pkl')

# 모델 로드 (필요할 때)
# vectorizer = joblib.load('tfidf_vectorizer.pkl')
# clf = joblib.load('random_forest_model.pkl')

# Feature로 선택된 단어 목록 출력
feature_names = vectorizer.get_feature_names_out()
print("Feature로 선택된 단어 목록:")
print(feature_names)

# 테스트 데이터 예측
X_test_tfidf = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_tfidf)

print(y_pred)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(report)
