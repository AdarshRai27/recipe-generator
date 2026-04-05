import json
import re
import nltk
import os
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from openai import OpenAI


# Load environment variables

load_dotenv()


# NLTK Setup (only if missing)

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')


# Load Dataset

def load_data(file_path="recipe_data.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Text Preprocessing

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    text = re.sub(r"\s+", " ", text.strip())

    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# Process Dataset

def process_recipes(recipe_data):
    processed_data = []

    for recipe in recipe_data:
        title = preprocess_text(recipe.get("title", ""))

        ingredients = [preprocess_text(i) for i in recipe.get("ingredients", [])]
        instructions = [preprocess_text(i) for i in recipe.get("instructions", [])]

        recipe_text = " ".join(ingredients + instructions)

        processed_data.append({
            "title": title,
            "text": recipe_text
        })

    return processed_data


# Save Processed Data

def save_processed_data(data, output_file="processed_data.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# OpenAI Setup

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Recipe Generator

def generate_recipe(ingredients, max_tokens=200, temperature=0.7):
    prompt = f"Create a detailed cooking recipe using these ingredients: {', '.join(ingredients)}."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful cooking assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].message.content.strip()


# CLI Interface

def generate_recipe_from_input():
    ingredients = input("Enter ingredients (comma-separated): ").split(",")

    max_tokens = input("Max tokens (default=200): ")
    max_tokens = int(max_tokens) if max_tokens else 200

    temperature = input("Temperature (default=0.7): ")
    temperature = float(temperature) if temperature else 0.7

    recipe = generate_recipe(ingredients, max_tokens, temperature)

    print("\n🍽️ Generated Recipe:\n")
    print(recipe)


# Main Execution

def main():
    try:
        data = load_data()
        processed = process_recipes(data)
        save_processed_data(processed)

        print("✅ Data processed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")

    generate_recipe_from_input()


if __name__ == "__main__":
    main()