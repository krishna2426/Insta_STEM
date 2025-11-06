# instaSTEM: ML-Powered Classifier for STEM Content on Instagram

**instaSTEM** is an early-stage machine learning project designed to identify and classify STEM (Science, Technology, Engineering, Mathematics) content on Instagram, starting with captions.

## Inspiration

I came up with this idea while watching a [video](https://youtu.be/3i59t2i7lhM?feature=shared) by [Neil deGrasse Tyson](https://en.wikipedia.org/wiki/Neil_deGrasse_Tyson), where he reviewed several TikToks featured in TikTok’s new STEM feed. That video made me realize how powerful a dedicated educational content filter can be — and how Instagram still doesn’t have anything like it.  
This sparked the idea to build a tool that can help surface meaningful STEM content on platforms where it often gets buried under entertainment overload.

## Vision

In a world filled with short-form entertainment, educational content often gets overlooked. TikTok has addressed this with its dedicated STEM feed — Instagram has not. This project aims to bridge that gap using machine learning, and I hope it does.

The long-term vision:
- Build a personalized STEM feed for Instagram
- Use intelligent content filtering to elevate science and tech creators
- Eventually expand to other platforms like YouTube Shorts and Reddit
- Enable filtering by subtopics (AI, physics, math, etc.)

## What’s Been Built So Far

- A working machine learning model trained to classify Instagram captions as either **STEM** or **Non-STEM**
- Dataset of 500+ labeled Instagram captions
- Text preprocessing (lowercasing, cleaning, hashtag retention)
- TF-IDF vectorizer with bi-grams
- Logistic Regression model
- Model and vectorizer saved as `.pkl` files for future app integration

## What’s Not Built Yet

- No user interface (web or mobile)
- No browser extension or automation
- No deployment
- No multi-platform support (yet)

## What's Next

- Build a minimal web app using Streamlit or Flask
- Develop a Chrome extension or mobile wrapper to apply filtering on Instagram
- Improve the model with transformer-based classifiers (e.g., BERT)
- Add support for sub-topic classification (e.g., Math, CS, Physics)
- Explore cross-platform STEM feed aggregation (Reddit, YouTube Shorts)

## Looking for Contributors

This is currently a solo project, and I’m looking for contributors to collaborate on the next phases.

You’re welcome to contribute if you’re into:
- Frontend development (React, Tailwind, Streamlit)
- Backend development (Flask, FastAPI)
- Browser extensions
- NLP and transformer models
- UI/UX design
- Dataset expansion or annotation
- Full-stack or MLOps

If you’re passionate about building tools that help make educational content more accessible, I’d love to work with you.  
Feel free to open an issue or reach out via [LinkedIn](https://www.linkedin.com/in/krishna-sharma-112366215).

## How to Run the Model Locally

```bash
# Clone the repo and install dependencies
pip install scikit-learn pandas

# In your Python script or notebook:
import pickle

with open("instaSTEM.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_caption(text):
    cleaned = text.lower()
    vec = vectorizer.transform([cleaned])
    return "STEM" if model.predict(vec)[0] == 1 else "Non-STEM"

# Example usage
predict_caption("Building a Python script to automate math problems")
