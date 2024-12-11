import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Ön işleme için lemmatizer ve stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Preprocess text data:
    - Lowercase conversion
    - Remove punctuation
    - Tokenize and lemmatize
    - Remove stop words
    """
    # Küçük harfe çevir, noktalama işaretlerini kaldır
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize et, lemmatize et ve durma kelimeleri çıkar
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


def build_similarity_matrix(data):
    """
    Build a similarity matrix based on preprocessed movie descriptions.
    """
    # Overview sütununu temizle
    data['Overview_Cleaned'] = data['Overview'].apply(preprocess_text)

    # TF-IDF vektörleştirme: İngilizce durma kelimeler çıkarılır, unigram ve bigram kullanılır
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['Overview_Cleaned'])

    # Cosine benzerlik matrisi
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix


def recommend_movies(movie_title, data, similarity_matrix, top_n=10):
    """
    Recommend movies similar to the given movie title based on preprocessed descriptions.
    """
    # Eğer film başlığı veri setinde yoksa hata mesajı döndür
    if movie_title not in data['Series_Title'].values:
        return f"Movie '{movie_title}' not found in dataset."

    # Verilen film başlığının indeksini al
    idx = data[data['Series_Title'] == movie_title].index[0]

    # İlgili filmin benzerlik skorlarını al
    similarity_scores = list(enumerate(similarity_matrix[idx]))

    # Benzerlik skorlarını büyükten küçüğe sırala
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # İlk 'top_n' filmi al (kendisi hariç)
    recommended_indices = [x[0] for x in sorted_scores[1:top_n + 1]]
    recommended_movies = data.iloc[recommended_indices]['Series_Title'].tolist()

    return recommended_movies
