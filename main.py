import gradio as gr
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    messages:Annotated[list, add_messages]


embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

groq_api_key = os.getenv("OPENAI_API_KEY")

if not groq_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

model = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0)
memory = MemorySaver()

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert in summarising a document that should say what the document is about. Document content: {question}. Give a summary in very short about 3 lines in a single paragraph<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

prompt_question = prompt | model | StrOutputParser()


final_summarisation = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert in summarising a document that should say what the document is about. Document content: {question}. Give a detailed summary that tells what the document is <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

final_prompt_question = final_summarisation | model | StrOutputParser()


task_route_prompt = PromptTemplate(
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

task_router_template = task_route_prompt | model | JsonOutputParser()

def identify_task(state: State): 
    state_messages = state['messages'][-1].content
    response = task_router_template.invoke(state_messages)
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


vectorstore = None
retriever = None 

def parse_content_list(page_content):
    split_text = re.split(r'\n(?![a-z])', page_content)
    return [i.replace('\n', '') for i in split_text]

def summarize_pdf_and_chat(pdf, chatbot):
    """Generates a PDF summary and adds it to the chatbot."""
    try:
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()
        # content_list = []
        all_page_summaries = ""
        individual_summary = ""
        content_list_doc = []
        for i in range(0, len(pages)):
            content = pages[i].page_content
            # content_list.append(pages[i])
            page_splitted_content = parse_content_list(pages[i].page_content)
            for cont in page_splitted_content:
                metadata = {'source' : pages[i].metadata['source'], 'page': pages[i].metadata['page']}
                page_content = cont
                content_list_doc.append(Document(page_content=page_content, metadata=metadata))

            individual_summary =  prompt_question.invoke({"question": content})
            all_page_summaries += individual_summary
            # print("Individual Summary is : ", individual_summary)

        vectorstore = Chroma.from_documents(documents=content_list_doc,embedding=embed_model,collection_name="local-rag")
        global retriever
        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={'score_threshold': 0.4})
        summary_result = final_prompt_question.invoke({"question": all_page_summaries}).strip()
        # generated_question = generate_question_chain.invoke({"summary":individual_summary}).strip()
        chatbot.append(("System", f"Summary of the PDF: {summary_result}"))
        # chatbot.append(("System", f"{generated_question}"))
        print("Retriever ready")
        return chatbot
    except Exception as e:
        chatbot.append(("System", f"Error processing the PDF: {e}"))
        return chatbot
    

def retrieve_context(state:State): 
    state_messages = state['messages']
    print("State message is : ", state_messages)
    response = retriever.invoke(state_messages[-1].content)
    print("Response from Retriever is : ", response)
    print(type(response))
    # return {"messages": response[0].page_content}
    context_str = ""
    for item in range(0, len(response)):
        context_str += 'Page Number : ' + str(response[item].metadata['page']+1) + ' Context :' + response[item].page_content + "\n"
    print("Context string is : ", context_str)
    return {"messages": context_str}

def user_interaction(chatbot, user_message):
    # print("Retriever is : ", retriever)
    """Handles user interaction with the chatbot."""
    chatbot.append(("User", user_message))
    # Placeholder for additional logic (e.g., answering questions about the PDF)

    response = app.invoke({"messages": user_message}, config)
    print("Final Response is : ", response)
    if response['messages'][-1].content == '':
        final_response = 'The question you asked is not related to my domain/can you rephrase the question'
    else:
        final_response = response['messages'][-1].content
    try:
        chatbot.append(("System", final_response))
    except Exception as e:
        chatbot.append(("System", f"Error during retrieval: {e}"))
    return chatbot, ""

workflow = StateGraph(State)

workflow.add_node("summarization", summarization)
workflow.add_node("control_Generation", control_Generation)
workflow.add_node("obligation_Generation", obligation_Generation)
workflow.add_node("question_answer", question_answer)
workflow.add_node("retrieve_context", retrieve_context)

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
workflow.add_edge("question_answer", "retrieve_context")
workflow.add_edge("retrieve_context", END)


app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

    
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("### Conversational PDF Summarization and Q&A")
    
    chatbot = gr.Chatbot(label="Conversation with PDF")
    pdf_file = gr.File(label="Upload your PDF", file_types=[".pdf"])
    user_input = gr.Textbox(label="Type your question here")
    
    with gr.Row():
        submit_button = gr.Button("Submit")
        clear_button = gr.Button("Clear Conversation")

    # Define Actions
    pdf_file.change(summarize_pdf_and_chat, inputs=[pdf_file, chatbot], outputs=chatbot)
    submit_button.click(user_interaction, inputs=[chatbot, user_input], outputs=[chatbot, user_input])
    clear_button.click(lambda: [], outputs=chatbot)  # Clears chatbot history


# Launch the App
demo.launch()
