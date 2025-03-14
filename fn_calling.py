
import instructor
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENROUTER_API_KEY"] = "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"

api_key = os.getenv("OPENROUTER_API_KEY")


def plot_x_y(x, y):
    """function to plot the x and y axes"""
    plt.plot(x, y)
    plt.show()

class FunctionCall(BaseModel):
    name : Literal["plot_x_y"] = "plot_x_y"
    arguments: "FunctionArguments"

class FunctionArguments(BaseModel):
    x: list = Field(..., description="x-axis values")
    y: list = Field(..., description="y-axis values")

FunctionCall.model_rebuild()

client = instructor.from_openai(
           OpenAI(base_url="https://openrouter.ai/api/v1",
           api_key=api_key),
           mode=instructor.Mode.JSON)

function_result = client.chat.completions.create(
    model="google/gemma-3-4b-it:free",
    response_model=FunctionCall,
    messages=[
        {"role": "system", "content": "you are helpfull assistant"},
        {"role": "user", "content": "Call the function plot_x_y with x as [12, 15, 18] and y as [20, 24, 29]"}
    ]
)

print(function_result)

if function_result.name == "plot_x_y":
    plot_x_y(**function_result.arguments.dict())
else:
    print("Unknown function call")
