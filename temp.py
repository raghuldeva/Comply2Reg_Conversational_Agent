import gradio as gr
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
import re
from langchain.schema.document import Document
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from typing import Annotated, Literal, Optional, Iterator, AsyncIterator
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    # question: str
    messages:Annotated[list, add_messages]

groq_api_key = os.getenv("OPENAI_API_KEY")

if not groq_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

model = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0)
memory = MemorySaver()

prompt = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert capable of identifying tasks like summarization, obligation generation, control generation, and question answering. Based on the input provided, determine the task:
    summarization: Condensing a document into a brief summary. Eg. 'Summarize this document: AI is transforming industries like healthcare, finance, and education by automating processes and improving efficiency'
    obligation_Generation: Extracting obligations from legal or formal documents. Eg. 'List the obligations of the vendor in this agreement: The vendor shall deliver all goods within 30 days of order placement and ensure product quality meets specified standards'
    control_Generation:  Creating policies or controls based on a request. Eg. 'Draft a policy to regulate employee access to sensitive data in the company.'
    question_answer: Answering factual questions based on the provided query. Eg. 'What is the Scope defined?'
    Give a choice like 'summarization', 'obligation_Generation', 'control_Generation', 'question_answer' based on the question. Return the a JSON with a single key 'task' and no premable or explaination.
    Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question"],
)

prompt_question = prompt | model | JsonOutputParser()

def identify_task(state: State): 
    state_messages = state['messages'][-1].content
    response = prompt_question.invoke(state_messages)
    print("State message is : ", state_messages)
    print("Response is : ", response['task'])
    print("The state is : ", state)
    if response["task"] == "summarization":
        return 'summarization'
    elif response["task"] == "control_Generation":
        return 'control_Generation'
    elif response["task"] == "obligation_Generation":
        return 'obligation_Generation'
    elif response["task"] == "question_answer":
        return 'question_answer'
    

summary_prompt = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert in summarisation task given a context. 
    The context for you is : {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question"],
)

summary_prompt_template = summary_prompt | model | StrOutputParser()

def summarization(state: State):
    print("ENTERED SUMMARISATION")
    state_messages = state['messages'][-2:]
    print("State msg summ : ", state_messages)
    response = summary_prompt_template.invoke(state_messages)
    return {'messages': response}


control_prompt = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert in control generation task given a context. 
    The context for you is : {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question"],
)

control_prompt_template = control_prompt | model | StrOutputParser()

def control_Generation(state: State):
    print("ENTERED CONTROL GENERATION")
    state_messages = state['messages'][-2:]
    print("Content for control genaration is : ", state_messages)
    response = control_prompt_template.invoke(state_messages)
    return {'messages': response}


obligation_prompt =  PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert in obligation generation task given a context. 
    The context for you is : {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question"],
)

obligation_prompt_template = obligation_prompt | model | StrOutputParser()

def obligation_Generation(state: State):
    print("ENTERED OBLIGATION GENERATION")
    state_messages = state['messages'][-2:]
    print("Content for obligation genaration is : ", state_messages)
    response = obligation_prompt_template.invoke(state_messages)
    return {'messages': response}

def question_answer(state:State):
    print("ENTERED QUESTION ANSWER")

workflow = StateGraph(State)

workflow.add_node("summarization", summarization)
workflow.add_node("control_Generation", control_Generation)
workflow.add_node("obligation_Generation", obligation_Generation)
workflow.add_node("question_answer", question_answer)

workflow.set_conditional_entry_point(
    identify_task,
    {
        "summarization":"summarization",
        "control_Generation": "control_Generation",
        "obligation_Generation":"obligation_Generation",
        "question_answer":"question_answer"
    }
)

workflow.add_edge("summarization", END)
workflow.add_edge("control_Generation", END)
workflow.add_edge("obligation_Generation", END)
workflow.add_edge("question_answer", END)

app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

# question = "Generate a summary for statement 'Financial institutions must implement adequate security measures to protect customer data against unauthorized access, alteration, or disclosure. This includes, but is not limited to, encryption of sensitive information, regular security audits, and adherence to industry standards for cybersecurity practices. Institutions are also required to provide training for employees on data protection protocols and monitor compliance with these standards. In cases of data breaches, institutions must notify affected individuals within a specified timeframe and report incidents to relevant authorities. Furthermore, customer consent is mandatory for data sharing, and institutions must provide customers with clear opt-out mechanisms. Failure to comply with these provisions may result in fines, legal action, or other penalties as determined by regulatory bodies.'"
# print(app.invoke({"messages": question}, config))

def user_interaction(chatbot, user_message):
    chatbot.append(("User", user_message))
    response = app.invoke({"messages": user_message}, config)
    try:
        chatbot.append(("System", f"{response['messages'][-1].content}"))
    except Exception as e:
        chatbot.append(("System", f"Error during retrieval: {e}"))
    return chatbot

with gr.Blocks(theme=gr.themes.Citrus) as demo:
    gr.Markdown("### Conversational PDF Summarization and Q&A")
    
    chatbot = gr.Chatbot(label="Conversation with PDF")
    user_input = gr.Textbox(label="Type your question here")
    
    with gr.Row():
        submit_button = gr.Button("Submit")
        clear_button = gr.Button("Clear Conversation")

    submit_button.click(user_interaction, inputs=[chatbot, user_input], outputs=chatbot)
    clear_button.click(lambda: [], outputs=chatbot) 

# Launch the App
demo.launch()