"""
Configuration settings for the Style Finder application.
"""

# Model and API configuration
# Any vision-capable OpenAI model works here. gpt-4o-mini is the cheap default;
# gpt-4o reads fine detail (fabric texture, hardware, small logos) noticeably better.
MODEL_ID = "gpt-4o-mini"

# Generation settings
# Low temperature keeps the model close to the retrieved item data instead of
# inventing garments; max_tokens is generous because one outfit can carry 11 items.
TEMPERATURE = 0.2
TOP_P = 0.6
MAX_TOKENS = 2000

# Image processing settings
IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# Default similarity threshold
SIMILARITY_THRESHOLD = 0.8

# Number of alternatives to return from search
DEFAULT_ALTERNATIVES_COUNT = 5
