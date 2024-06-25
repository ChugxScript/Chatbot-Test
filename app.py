from langchain_community.llms import Ollama 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st

llm = Ollama(model="llama3")

# Prompt template
prompt = PromptTemplate(
    template = """
      You are a world class business development representative.
      I will share a prospect's message with you and you will give me the best answer that
      I should send to this prospect based on past best practies,
      and you will follow ALL of the rules below:

      1/ Response should be very similar or even identical to the past best practies,
      in terms of length, ton of voice, logical arguments and other details

      2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

      3/ If the prospect's message is relevant to the bot_core,
      then try to explain the details to the prospect \n

      Below is a message I received from the prospect:
      {message}

      Here is a list of best practies of how we normally respond to prospect in similar scenarios:
      {best_practice}

      Here is the bot_core:
      {BOT_CORE}

      Please write the best response that I should send to this prospect:
      """,
    input_variables = ["message", "best_practice", "BOT_CORE"],
)

chain = prompt | llm | JsonOutputParser()

def get_bot_response(prompt):
    # Load the CSV file into a DataFrame
    df = pd.read_csv("./bot_data/bot_response.csv")

    # Extract the customer messages from the DataFrame
    customer_messages = df['Customer message'].tolist()

    # Combine the customer messages with the prompt for vectorization
    corpus = customer_messages + [prompt]

    # Vectorize the customer messages and the prompt using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate cosine similarity between the prompt and all customer messages
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Find the indices of the top 3 most similar customer messages
    top_3_indices = similarity_scores.argsort()[-3:][::-1]

    # Return the top 3 most similar documents
    top_3_documents = df.iloc[top_3_indices]
    return top_3_documents

def get_bot_core():
    BOT_PATH = './bot_data/bot_core.txt'
    with open(BOT_PATH, 'r') as f:
        bot_core = f.read()
    return bot_core


st.title("CHATBOT🤖")
user_prompt = st.text_area("Enter your prompt:")

if st.button("ENTER"):
    if user_prompt:
        with st.spinner("Generating response..."):
            best_practice_docs = get_bot_response(user_prompt)
            best_practice = "\n\n".join(
                [f"Customer message: {row['Customer message']}\nOur response: {row['Our response']}" for index, row in best_practice_docs.iterrows()]
            )
            BOT_CORE = get_bot_core()
            message = user_prompt
            st.write_stream(
                llm.stream(chain.invoke({"message": message, "best_practice": best_practice, "BOT_CORE": BOT_CORE,}),
                stop=['<|eot_id|>'])
              )