from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from fewshot_example import examples
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from functions import *
import streamlit as st
import os

#set open api key
openai_api_key=os.environ.get("OPENAI_API_KEY")

with st.container():
    st.markdown("""
# Prompt Generator 
## Genertae Prompt From The Question
In this streamlit application, we are demonstrating how to build an Interactive prompt generator.

We've utilized Langchain, a powerfull tool that aids in the generation of applications using language models. Langchain provides a set of components that streamline the process of creating and formatting prompts for language models. and this application a straightforward implementation of these components.

Here's how this interactive prompts generator operate:
- Users enter an initial prompt, which serves as the seed for the language model's creative process.
- The application then uses langchain to create a more refined and contextualized prompt, drawing from a set of predefined examples.
- These examples are selected based on their semantic similarity to the user's initial prompt, ensuring the output is relevant and focused.
- The final, improved prompt is then displayed on the user interface.

This interactive generator is part of the learning seies, where we explore the diffrent facets of using langchain for language model prompt generation.
Find the code on clicking here : [Github]("https://github.com/AKASH4148")                            
"""
                )
with st.container():
    st.markdown("""
                ## Enter intial Prompt here: 
""")
initial_prompt=st.text_area(label="Prompt Input", label_visibility="collapsed", placeholder="Generate a work ot schedule", key="prompt_input")
if initial_prompt:
    if initial_prompt== "empty":
        st.write("Please enter a valid prompt")
        st.stop()
    else:
        with st.spinner("Generating prompt...."):
            #creating a example selector, this is used to select examples that are similar to the input prompt.
            example_selector=create_example_sel(examples, openai_api_key, number_of_examples=1)

            #create a prompt template. this is used to format the prompt.
            prompt_template=create_prompt_template(example_selector)

            #format the prompt with the givem initial prompt provided by the user 
            formatted_prompt=format_prompt(prompt_template, initial_prompt)

            #Initialize the language model, this is the model that will generate the output
            llm=initialize_llm(openai_api_key, model_name="gpt-3.5-turbo", temperature=0.2)

            #initialize the llm chain
            llm_chain=initialize_llm_chain(llm, formatted_prompt)

            #genrate the imporoved prompt
            improved_prompt=generate_improved_prompt(llm_chain)
            st.write(improved_prompt)
