import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import openai
import os

nltk.download('stopwords')
nltk.download('wordnet')

# Load the recipe data from the JSON file
with open("recipe_data.json", "r") as f:
    recipe_data = json.load(f)

# Define a function to preprocess the recipe text
def preprocess_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = " ".join(word for word in text.split() if word not in stop_words)
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split())
    return text

# Extract the ingredient and instruction data from each recipe
processed_data = []
for recipe in recipe_data:
    # Extract the recipe title and preprocess it
    title = preprocess_text(recipe["title"])
    # Extract the ingredient and instruction data and preprocess it
    ingredients = [preprocess_text(i) for i in recipe["ingredients"]]
    instructions = [preprocess_text(i) for i in recipe["instructions"]]
    # Combine the ingredient and instruction data into a single string
    recipe_text = " ".join(ingredients) + " " + " ".join(instructions)
    # Add the processed data to the output list
    processed_data.append({"title": title, "text": recipe_text})

# Save the processed data to a JSON file
with open("processed_data.json", "w") as f:
    json.dump(processed_data, f)

# Initialize the OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a function to generate new recipes using GPT-3
def generate_recipe(ingredients, max_length=200, temperature=0.5):
    prompt = "Here's a recipe you can make with " + ", ".join(ingredients) + ":\n\n"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=max_length,
        temperature=temperature,
        n=1,
        stop=None,
        timeout=30,
    )
    recipe = response.choices[0].text
    return recipe.strip()

# Define a function to generate new recipes based on user input
def generate_recipe_from_input():
    # Get user input for ingredients and other criteria
    ingredients = input("Enter the ingredients you have: ").split(",")
    max_length = input("Enter the maximum recipe length (default=200): ")
    if max_length:
        max_length = int(max_length)
    else:
        max_length = 200
    temperature = input("Enter the recipe generation temperature (default=0.5): ")
    if temperature:
        temperature = float(temperature)
    else:
        temperature = 0.5
    
    # Generate a new recipe based on the user input
    recipe = generate_recipe(ingredients, max_length=max_length, temperature=temperature)
    print(recipe)

# Generate a new recipe based on user input
generate_recipe_from_input()
