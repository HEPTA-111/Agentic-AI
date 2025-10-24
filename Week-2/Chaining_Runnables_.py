from langchain_core.prompts import PromptTemplate
# We changed this line:
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# First chain generates questions
question_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate 3 questions about {topic}:")

# Second chain generates answers based on questions
answer_prompt = PromptTemplate(
    input_variables=["questions"],
    template="Answer the following questions:\n{questions}\n You response should contain the question and the answer to it.")

# We changed this line:
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

# Output parser to convert model output to string
output_parser = StrOutputParser()

# Build the question generation chain
question_chain = question_prompt | llm | output_parser

# Build the answer generation chain
answer_chain = answer_prompt | llm | output_parser

# Define a function to create the combined input for the answer chain
def create_answer_input(output):
    return {"questions": output}

# Chain everything together
qa_chain = question_chain | create_answer_input | answer_chain

# Run the chain
result = qa_chain.invoke({"topic": "artificial intelligence"})

# Added this so you can see the final output!
print(result)