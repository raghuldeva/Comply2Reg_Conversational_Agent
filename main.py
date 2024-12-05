import gradio as gr
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

class State(TypedDict):
    messages:Annotated[list, add_messages]


embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

groq_api_key = os.getenv("OPENAI_API_KEY")

if not groq_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

model = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert in summarising a document that should say what the document is about. Document content: {content}. Give a summary in very short about 3 lines in a single paragraph<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["content"],
)

prompt_question = prompt | model | StrOutputParser()


final_summarisation = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert in summarising a document that should say what the document is about. Document content: {content}. Give a detailed summary that tells what the document is <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["content"],
)

final_prompt_question = final_summarisation | model | StrOutputParser()

# generate_question_prompt =  PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert in generating a single question from the summary. Summary: {summary}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
#     input_variables=["summary"],
# )

# generate_question_chain = generate_question_prompt | model | StrOutputParser()

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

            individual_summary =  prompt_question.invoke({"content": content})
            all_page_summaries += individual_summary
            print("Individual Summary is : ", individual_summary)

        vectorstore = Chroma.from_documents(documents=content_list_doc,embedding=embed_model,collection_name="local-rag")
        global retriever
        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={'score_threshold': 0.4})
        summary_result = final_prompt_question.invoke({"content": all_page_summaries}).strip()
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
    response = retriever.invoke(state_messages[0].content)
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

    response = app.invoke({"messages": user_message})
    print("Final Response is : ", response)
    if response['messages'][-1].content == '':
        final_response = 'The question you asked is not related to my domain/can you rephrase the question'
    else:
        final_response = response['messages'][-1].content
    try:
        chatbot.append(("System", f"{final_response}"))
    except Exception as e:
        chatbot.append(("System", f"Error during retrieval: {e}"))
    return chatbot

workflow = StateGraph(State)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_edge(START, "retrieve_context")
workflow.add_edge("retrieve_context", END)
app = workflow.compile()

    
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
    submit_button.click(user_interaction, inputs=[chatbot, user_input], outputs=chatbot)
    clear_button.click(lambda: [], outputs=chatbot)  # Clears chatbot history


# Launch the App
demo.launch()
