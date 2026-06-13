# Yelp Hotel Competitive Analysis

Interactive analysis of Yelp hotel reviews using natural language processing and topic modeling to identify competitive positioning drivers and customer sentiment patterns.

# Overview

This project analyzes 950+ Yelp reviews of Chicago-area hotels to uncover what drives customer satisfaction, identify competitive advantages, and inform market-entry or positioning strategy. Uses Latent Dirichlet Allocation (LDA) topic modeling to extract underlying themes from unstructured review text, then maps themes to customer sentiment and rating patterns.

# Key Findings:

- Location and views dominate positive sentiment (Lake Michigan waterfront, Navy Pier proximity)
- Service quality and room comfort are polarizing factors — drive both 5-star and 1-star ratings
- Hidden costs (parking, WiFi, resort fees) create negative surprises despite strong amenities
- Brand differentiation opportunity: competitor positioning as "trendy/nightlife-focused" alienates family travelers and business guests

# Features

## Data & Analysis

- 950+ Yelp reviews across rating spectrum (1–5 stars)
- LDA topic modeling — unsupervised discovery of 6-8 latent themes
- Sentiment mapping — linking themes to positive/negative review language
- Rating correlation — which themes correlate with high vs. low ratings

## Interactive Insights

- Topic distribution across review corpus
- Sentiment breakdown by theme
- Rating heatmaps by theme and customer segment
- Competitive positioning matrix

# Use Cases

- Hotel expansion: Which neighborhood characteristics and guest experience dimensions drive loyalty in underperforming markets?
- Competitive intelligence: What are guests saying about your brand vs. competitors?
- Operations prioritization: Which service gaps (room size, WiFi, parking transparency) are costing you stars?

# Technical Stack

- Language Processing: Python (NLTK, spacy), TF-IDF vectorization
- Topic Modeling: Scikit-learn (LDA), Gensim
- isualization: matplotlib, seaborn, pyLDAvis
- Data: Yelp review corpus (950+ reviews)

# About

Built as an applied NLP project for market analysis. Demonstrates ability to extract business intelligence from unstructured customer feedback, perform unsupervised learning at scale, and translate technical findings into strategic recommendations.
