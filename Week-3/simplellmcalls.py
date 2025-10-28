from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import yaml
import os
from dotenv import load_dotenv


load_dotenv()  
llm = ChatGroq (
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY") 
)

publication_content = """
Title: One Model, Five Superpowers: The Versatility of Variational Auto-Encoders

TL;DR
Variational Auto-Encoders (VAEs) are versatile deep learning models with applications in data compression, noise reduction, synthetic data generation, anomaly detection, and missing data imputation. This publication demonstrates these capabilities using the MNIST dataset, providing practical insights for AI/ML practitioners.

Introduction
Variational Auto-Encoders (VAEs) are powerful generative models that exemplify unsupervised deep learning. They use a probabilistic approach to encode data into a distribution of latent variables, enabling both data compression and the generation of new, similar data instances.
[rest of publication content... truncated for brevity]
"""

# Same question, grounded in publication context
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content=f"""
Based on this publication: {publication_content}

What are variational autoencoders and list the top 5 applications for them as discussed in this publication.
""")
]

response = llm.invoke(messages)
print(response.content)