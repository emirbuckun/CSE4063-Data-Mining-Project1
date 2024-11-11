import ssl
from ucimlrepo import fetch_ucirepo

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# fetch dataset
recipe_reviews_and_user_feedback = fetch_ucirepo(id=911)

# data (as pandas dataframes)
X = recipe_reviews_and_user_feedback.data.features
y = recipe_reviews_and_user_feedback.data.targets

# metadata
print(recipe_reviews_and_user_feedback.metadata)

# variable information
print(recipe_reviews_and_user_feedback.variables)
