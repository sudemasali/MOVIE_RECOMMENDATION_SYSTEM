# Movie Recommendation System

An AI-powered movie recommendation system built with Flask and NLP techniques.

The project recommends similar movies by analyzing movie descriptions using content-based filtering and cosine similarity.

---

## Project Overview

This project was developed to explore how recommendation systems work in modern digital platforms.

The system processes movie data, analyzes textual similarities and generates personalized movie recommendations through an API-based architecture.

---

## Features

- Movie recommendation engine
- NLP-based text processing
- Content-based filtering
- Cosine similarity algorithm
- REST API architecture
- Data preprocessing pipeline

---

## Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- NLP
- TF-IDF Vectorization
- Cosine Similarity

---

## Architecture

The project consists of:

- Data preprocessing module
- Recommendation engine
- Flask REST API
- Similarity analysis pipeline

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| POST | `/process-data` | Cleans and preprocesses raw movie data |
| POST | `/recommendd` | Returns movie recommendations |

---

## Recommendation Logic

The recommendation engine works by:

1. Cleaning and preprocessing movie descriptions
2. Applying TF-IDF vectorization
3. Calculating cosine similarity scores
4. Finding movies with similar textual content

---

## Business Perspective

Recommendation systems are widely used in digital platforms to improve user experience and increase user engagement.

This project demonstrates the core logic behind AI-powered personalization systems used in modern streaming platforms.

---

## Future Improvements

Possible future enhancements include:

- Frontend user interface
- Advanced recommendation algorithms
- User-based collaborative filtering
- Hybrid recommendation systems
- Real-time recommendation engine
- Deployment to cloud platforms

---

## Setup

```bash
pip install pandas numpy scikit-learn nltk flask flask-cors
```

Run the application:

```bash
python app.py
```

---

## Author

Sude Maşalı,Batuhan Çelikbaş,Rumeysa Tüysüz,Berna Tan

Computer Engineer interested in AI, Business & Technology
