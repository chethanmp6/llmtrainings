#--------------------------------------------------------------------------
#With_structured_output example using Pydantic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

# Define your output structure
class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    rating: int = Field(description="Rating from 1-10")
    summary: str = Field(description="Brief summary of the review")
    recommend: bool = Field(description="Whether you'd recommend it")

# Create the model with structured output
llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(MovieReview)

# Invoke and get a Pydantic object back
result = structured_llm.invoke("Review the movie Inception")

print(result.title)      # "Inception"
print(result.rating)     # 9
print(result.recommend)  # True
print(type(result))      # <class 'MovieReview'>

#--------------------------------------------------------------------------

#Nested Structured Output example using Pydantic
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    email: Optional[str] = None
    address: Address
    hobbies: List[str] = Field(default_factory=list)

structured_llm = llm.with_structured_output(Person)

result = structured_llm.invoke(
    "Extract info: John Smith, 32 years old, lives at 123 Main St, Boston, USA. "
    "Enjoys hiking and photography. Email: john@example.com"
)

print(result.name)              # "John Smith"
print(result.address.city)      # "Boston"
print(result.hobbies)           # ["hiking", "photography"]

#--------------------------------------------------------------------------

#Enum Field example using Pydantic
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class SentimentAnalysis(BaseModel):
    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    reasoning: str

structured_llm = llm.with_structured_output(SentimentAnalysis)

result = structured_llm.invoke("Analyze: 'This product exceeded my expectations!'")
print(result.sentiment)    # Sentiment.POSITIVE
print(result.confidence)   # 0.95

#--------------------------------------------------------------------------