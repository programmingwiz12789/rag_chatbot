import os
import ollama
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

def create_vector_db(query):
    try:
        search_tool = TavilySearchResults(max_results=5)
        search_results = search_tool.invoke(query)
        docs = [result['content'] for result in search_results if 'content' in result]
        if not docs:
            return None, 'No relevant content found'
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.create_documents(docs)
        embeddings = OllamaEmbeddings(model='nomic-embed-text')
        vector_db = Chroma.from_documents(documents=splits, embedding=embeddings)
        return vector_db, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    try:
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def rag(question):
    vector_db, error = create_vector_db(question)
    if error:
        return error
    retriever = vector_db.as_retriever()
    retrieved_docs = retriever.invoke(question)
    formatted_context = '\n\n'.join(doc.page_content for doc in retrieved_docs)
    return llm(question, formatted_context)

def get_answer(question):
    if not question:
        return 'Please enter a question.'
    return rag(question)

os.environ['TAVILY_API_KEY'] = '[YOUR_TAVILY_API_KEY]'

interface = gr.Interface(
    inputs=gr.Textbox(
        lines=2,
        placeholder='Enter your question here (e.g. What is RAG?)',
    ),
    outputs=gr.TextArea(),
    title='RAG Chatbot',
    description='Ask any question, and I will search the web to answer it!',
    fn=get_answer
)

interface.launch(debug=True)