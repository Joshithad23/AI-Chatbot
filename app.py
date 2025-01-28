import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

recognizer = sr.Recognizer()
parser = StrOutputParser()
engine = pyttsx3.init()

llm = ChatGoogleGenerativeAI(
    temperature=0.4,
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def SpeakText(command):
    """
    Converts text to speech and plays it.
    """
    engine.say(command)
    engine.runAndWait()

def audio():
    """
    Listens for audio input, converts speech to text, and returns the recognized text.
    """
    with sr.Microphone() as source:
        print("Listening for audio input...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio_data = recognizer.listen(source, timeout=20, phrase_time_limit=20)
            user_input = recognizer.recognize_google(audio_data)
            return user_input.lower()
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError as e:
            return f"Speech recognition error: {str(e)}"

@tool
def search_tool(query: str) -> str:
    """Simulates a search engine tool that returns results for a query."""
    return f"Search results for: {query}"

tools = [
    Tool(name="Search", func=search_tool, description="Use this tool for searching the web."),
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",
    verbose=True,
    memory=memory
)

st.set_page_config(page_title="AI Chatbot with Speech Features", layout="centered")
st.title("\U0001F916 AI Chatbot with Speech Features")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  

if "user_input" not in st.session_state:
    st.session_state.user_input = ""  


def display_chat_history():
    """Displays the chat history dynamically."""
    for message, sender in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f"**\U0001F464 You:** {message}")
        else:
            st.markdown(f"**\U0001F916 Bot:** {message}")


with st.container():
    st.subheader("Chat with the AI")
    user_input = st.text_input("Type your message or use speech-to-text:", value=st.session_state.user_input, key="input", max_chars=200, label_visibility="collapsed")

    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    with col1:
        if st.button("Send"):
            if user_input.strip():
                st.session_state.chat_history.append((user_input, "user"))
                try:
                    response = agent.run(user_input)
                    st.session_state.chat_history.append((response, "bot"))
                    st.session_state.user_input = ""
                except Exception as e:
                    st.session_state.chat_history.append((f"An error occurred: {str(e)}", "bot"))
            else:
                st.warning("Please type a message before sending.")

    with col2:
        if st.button("Voice Over"):
            st.write("Listening... Say something.")
            user_input = audio()
            st.session_state.chat_history.append((user_input, "user"))
            try:
                response = agent.run(user_input)
                st.session_state.chat_history.append((response, "bot"))
            except Exception as e:
                st.session_state.chat_history.append((f"An error occurred: {str(e)}", "bot"))

    with col3:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.user_input = ""

st.divider()
st.subheader("Chat History")
with st.container():
    display_chat_history()
