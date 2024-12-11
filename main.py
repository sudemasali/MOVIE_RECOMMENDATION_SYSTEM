from data_processing import load_data, preprocess_data
from model import build_similarity_matrix, recommend_movies


def main():
    # Veriyi yükle ve işle
    data_file = "data/raw/imdb_top_1000.csv"
    data = load_data(data_file)

    # Veri ön işleme (opsiyonel)
    data = preprocess_data(data)  # Genre gibi sütunları işler

    # Benzerlik matrisi oluştur
    similarity_matrix = build_similarity_matrix(data)

    # Kullanıcıdan film başlığı al ve öneriler yap
    movie_title = input("Enter a movie title: ")
    recommendations = recommend_movies(movie_title, data, similarity_matrix)

    if isinstance(recommendations, list):
        print(f"\nMovies similar to '{movie_title}':")
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie}")
    else:
        print(recommendations)


if __name__ == "__main__":
    main()
