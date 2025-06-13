import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import openai
import numpy as np

# Load dataset and index
df = pd.read_csv("wolaytta_dataset.csv")
index = faiss.read_index("wolaytta_faiss.index")
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Improved context retrieval with chunking and threshold
def retrieve_context(query, top_k=5, similarity_threshold=0.5):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k=top_k)
    retrieved = []
    for dist, idx in zip(D[0], I[0]):
        if dist < similarity_threshold:
            retrieved.append(df.iloc[idx]["wolaytta"])
    return "\n".join(retrieved) if retrieved else None

def ask_gpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300,
        temperature=0.7,
        n=1,
        stop=None,
    )
    return response.choices[0].message['content'].strip()

def rag_gpt_reply(user_input, user_prompt):
    context = retrieve_context(user_input)
    if context:
        system_content = f"Use the following Wolaytta phrases as context to answer questions:\n{context}"
    else:
        system_content = "No relevant Wolaytta context found. Answer based on your general knowledge."

    if user_prompt:
        system_content += f"\n\nUser custom prompt:\n{user_prompt}"

    system_message = {"role": "system", "content": system_content}
    conversation = st.session_state.messages + [system_message, {"role": "user", "content": user_input}]

    reply = ask_gpt(conversation)
    return reply

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about Wolaytta language using provided context."}
    ]
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = ""

st.title("Wolaytta RAG + GPT Chatbot 🤖")

user_prompt = st.text_area(
    "Add your custom system prompt (optional):",
    value=st.session_state.custom_prompt,
    height=100,
    help="This prompt guides the assistant's behavior during the conversation."
)
st.session_state.custom_prompt = user_prompt

col1, col2 = st.columns(2)
with col1:
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions about Wolaytta language using provided context."}
        ]
        st.experimental_rerun()

with col2:
    if st.button("Reset Prompt"):
        st.session_state.custom_prompt = ""
        st.experimental_rerun()

user_input = st.text_input("You:", key="input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = rag_gpt_reply(user_input, st.session_state.custom_prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.input = ""

for msg in st.session_state.messages[1:]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Bot:** {msg['content']}")
    else:
        st.markdown(f"*System:* {msg['content']}")
#display coversation history 