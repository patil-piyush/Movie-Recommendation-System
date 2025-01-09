# Movie Recommendation System

Welcome to the **Movie Recommendation System** project! This system leverages advanced data cleaning, visualization, and machine learning techniques to recommend movies to users based on their preferences.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Workflow](#workflow)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Datasets](#datasets)
7. [Code Explanation](#code-explanation)
    - [Data Cleaning](#data-cleaning)
    - [Data Visualization](#data-visualization)
    - [Model Building](#model-building)
8. [Techniques Used](#techniques-used)
9. [Results](#results)
10. [Future Work](#future-work)
11. [Contributors](#contributors)

---

## Project Overview
This project builds a **Movie Recommendation System** using machine learning techniques. It combines **collaborative filtering**, **content-based filtering**, and **hybrid approaches** to provide accurate recommendations. The system also visualizes data trends to enhance understanding and interpretation.

---

## Workflow
1. **Data Cleaning**: Preprocess raw data to remove missing values and duplicates.
2. **Data Visualization**: Perform univariate, bivariate, and multivariate analysis to explore relationships and trends in the dataset.
3. **Model Building**: Implement algorithms for collaborative, content-based, and hybrid filtering.
4. **Recommendation Generation**: Use trained models to recommend movies based on user preferences.
5. **Deployment**: Create an interactive web interface using **Gradio**.

![Workflow](https://via.placeholder.com/800x400.png?text=Workflow+Diagram)

---

## Features
- **Collaborative Filtering**: Recommends movies based on user interactions.
- **Content-Based Filtering**: Recommends movies with similar content (e.g., genres).
- **Hybrid Filtering**: Combines collaborative and content-based techniques for better accuracy.
- **Interactive Interface**: Provides an easy-to-use web application for recommendations.

---

## Project Structure
```
Movie-Recommendation-System/
|-- datasets/
|   |-- movie.csv
|   |-- tag.csv
|   |-- rating.csv
|-- data_cleaning.ipynb
|-- data_visualization.ipynb
|-- model.ipynb
|-- app.py
|-- README.md
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Movie-Recommendation-System.git
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   python app.py
   ```

---

## Datasets
The project uses the following datasets:
- **Movies**: Contains information about movies (e.g., title, genres).
- **Tags**: Includes user-generated tags for movies.
- **Ratings**: Stores user ratings for movies.

---

## Code Explanation

### 1. Data Cleaning
- **Purpose**: Clean and preprocess raw data to ensure consistency.
- **Steps**:
  - Handle missing values in `tag.csv`.
  - Remove duplicate `movieId` entries.
  - Merge `movie.csv`, `tag.csv`, and `rating.csv` to create the final dataset.
  - Save the cleaned dataset as `final_data.csv`.

### 2. Data Visualization
- **Purpose**: Explore data relationships and trends.
- **Steps**:
  - Univariate Analysis: Analyze single variables (e.g., rating distribution).
  - Bivariate Analysis: Analyze relationships between two variables (e.g., rating vs. genres).
  - Multivariate Analysis: Explore complex relationships using pair plots and correlation matrices.

### 3. Model Building
#### Collaborative Filtering:
- Uses the **SVD** algorithm to recommend movies based on user interaction.
- Example: If User A likes Movie X, and User B likes Movie X and Movie Y, Movie Y may be recommended to User A.

#### Content-Based Filtering:
- Extracts features like genres using **TF-IDF** and calculates similarities using **cosine similarity**.
- Example: If Movie A and Movie B share similar genres, and a user likes Movie A, Movie B may be recommended.

#### Hybrid Filtering:
- Combines predictions from collaborative and content-based models for better accuracy.

### Deployment
- **Gradio**: An interactive interface where users can select a movie to get recommendations.

---

## Techniques Used
- **Data Cleaning**: Pandas, Numpy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: SVD, TF-IDF, Cosine Similarity
- **Deployment**: Gradio for web interface

---

## Results
- Accurate and personalized movie recommendations.
- Insights into user behavior and movie trends through visualization.

---

## Future Work
- Enhance hybrid filtering by weighting collaborative and content-based predictions adaptively.
- Add support for real-time data updates.
- Incorporate NLP techniques for tag-based analysis.
- Extend the system to recommend TV shows and documentaries.

---

## Contributors
- **Piyush Patil** - Developer

Feel free to contribute to this project by submitting issues or pull requests!

---

Thank you for exploring this project! If you found it interesting, don’t forget to give it a star ⭐ on [GitHub](https://github.com/patil-piyush/Movie-Recommendation-System).

