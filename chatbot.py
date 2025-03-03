import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter ## for splitting the text into chunks
from langchain_community.document_loaders import WebBaseLoader ## for loading the text from website
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from langgraph.checkpoint.memory import MemorySaver

from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,END,START
import uuid
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

## ignoring the warnings
import warnings
warnings.filterwarnings('ignore')
## loading the secret variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")

## loading the llm and embeddings
llm = ChatOpenAI(model = "gpt-3.5-turbo",api_key=api_key)
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## creating the retriever for RAG system
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]
docs = [WebBaseLoader(url).load() for url in urls]
## my content is in nested list
## but i want my content in single list only
docs_list = [item for sublist in docs for item in sublist]
## i have my content in a single list
## now my next job is to split this text into chunks
## first of all i will be creating the RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 200,chunk_overlap = 100
)
docs_split = text_splitter.split_documents(docs_list)
## now i have my content in form of chunks
## now i will be converting these chunks into word vector embeddings
## and will store in a vector database
vectorstore = Chroma.from_documents(
    documents=docs_split,
    embedding=embeddings,
    collection_name="rag-chroma"
)
## creating retriever on top of this retriever
retriever = vectorstore.as_retriever()
tool = create_retriever_tool(retriever,
                             "retrieve_blog_posts",
                             "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs",)
tools = [tool]


class GradeDocuments(BaseModel):
    "Relevance score 'yes' or 'no'" 
    binary_score:str = Field(
        description = "Documents are relevant to the question 'yes' or 'no'"
    )

llm_with_structured_output1 = llm.with_structured_output(GradeDocuments)
system_grade = """ You are a document grader.Given the question and given to you.
your job is to find out either these documents are relevant to question or not.
if the question is relevant to documents return binary_score = 'yes' 
and if the question is not relevant to documents then return binary_score = 'no'
                                """  
prompt_document = ChatPromptTemplate.from_messages(
[
    ("system",system_grade),
    ("human","documents:{documents} \n question:{question}")
]
)
document_grader = prompt_document | llm_with_structured_output1

from langchain_core.output_parsers import StrOutputParser
prompt_rag = hub.pull("rlm/rag-prompt")

rag_chain = prompt_rag | llm



## formatting the output of the system 
## that i want my output in this format only
class GradeQuestion(BaseModel):
    "Grades question with documents" 
    binary_score:str = Field(
        description ="Relevance score 'yes' or 'no' "
    )
llm_with_structured2 = llm.with_structured_output(GradeQuestion)
system_question="""an Intelligent AI system.Given the question and generation to you.
with your intelligence you need to find out either this question is related to this generation or not.
if the question is related to the generation return binary_score = 'yes'
and if the question is not related to generation then return binary_score = 'no'
                  
                            """
prompt_question = ChatPromptTemplate.from_messages(
    [("system",system_question),
     ("human","question:{question} \n generation:{generation} ")]
)

answer_grader = prompt_question | llm_with_structured2


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system_hallu = """You are a grader checking if an LLM generation is grounded in or supported by a set of retrieved facts.  
Give a simple 'yes' or 'no' answer. 'Yes' means the generation is grounded in or supported by a set of retrieved the facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_hallu),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader = hallucination_prompt | structured_llm_grader

system_rewrite = """You are a question re-writer that converts an input question into a better optimized version for vector store retrieval document.  
You are given both a question and a document.  
- First, check if the question is relevant to the document by identifying a connection or relevance between them.  
- If there is a little relevancy, rewrite the question based on the semantic intent of the question and the context of the document.  
- If no relevance is found, simply return this single word "question not relevant." dont return the entire phrase 
Your goal is to ensure the rewritten question aligns well with the document for better retrieval.    
Note this thing if you find question is not releated to documents then in that case just return "question not relevant"
DO not give the explaination if the question is not relevant to documents
      

         """
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewrite),
        (
            "human","""Here is the initial question: \n\n {question} \n,
             Here is the document: \n\n {documents} \n ,
             Formulate an improved question. if possible other return 'question not relevant'."""
        ),
    ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()

class AgentState(TypedDict):
    question:str
    generation:str
    documents:List[str]
    filter_documents:List[str]
    unfilter_documents:List[str]


## now creating the chatbot class

class chatbot:
    def __init__(self):
        pass
    def retrieve(self,state:AgentState):
        question = state.get("question",None)
        if question is None:
            print("question not found")
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}
    
    def grade_document(self,state:AgentState):
        question = state["question"] ## question
        documents = state["documents"] 
        filtered_docs = [] ## all the filtered documents will be stored in this list only
        unfiltered_docs = [] ## all the unfiltered documents will be stored in this list only

        for doc in documents:
            binary_score = document_grader.invoke({"question":question,"documents":doc}).binary_score
            if binary_score == 'yes':
                filtered_docs.append(doc)
            elif binary_score == 'no':
                unfiltered_docs.append(doc)
        ## now i have both filtered_doc and unfiltered_doc
        ## now if there is any unfiltered doc i will not return any filtered doc
        if len(unfiltered_docs) >1:
            return {"question":question,"documents":documents,"unfiltered_documents":unfiltered_docs,"filtered_documents":[]} ## in this case i am getting binary_score=no for any document
        else:
            return {"question":question,"documents":documents,"unfiltered_documents":[],"filtered_documents":filtered_docs} ## here i am not getting binary_score = 'no' for any document
    def decide_to_generate(self,state:AgentState):
        question = state["question"]
        documents = state["documents"]
        unfiltered_documents = state["unfiltered_documents"]
        filtered_documents = state["filtered_documents"]
        
        ## taking the condition where there are some documents whose binary_score was no
        if unfiltered_documents:
            return "transform_query" 
        if filtered_documents:
            return "generate" 
    def generate(self,state:AgentState):
        question = state["question"]
        documents = state["documents"]
        response = rag_chain.invoke({"question":question,"context":documents})
        return {"question":question,"documents":documents,"generation":response}
    ## suppose we went to first case
    ## now we need to rewrite the query based on the previous query intent and retrieved documents context
    def transform_query(self,state:AgentState):
        question = state["question"]
        documents = state["documents"]
    

        question = question_rewriter.invoke({"question":question,"documents":document_grader})
        ## we passed the user question to question rewriter
        ## now there are two possibilities
        ## first possibility is that we will get the new generated question
        if question == 'question not relevant': ## if this case arises we will move to end and we also need to give generation
            return {"question":question,"documents":documents,"generation":"question not at all relevant"}
        else:
            return {"question":question,"documents":documents}
    
    def decide_to_generate_after_transform(self,state:AgentState):
        question = state["question"]
        if question == 'question not relevant':
            return "query_not_at_all_relevant" 
        else:
            return "Retriever" 
        
    def grade_generation_vs_documents_and_question(self,state:AgentState):
        question = state["question"] ## user question
        documents = state["documents"] ## retrieved documents
        generation = state["generation"] ## generation

        binary_score = hallucination_grader.invoke({"documents":documents,"generation":generation}).binary_score
        if binary_score == 'yes':
            binary_score = answer_grader.invoke({"generation":generation,"question":question}).binary_score
            if binary_score == 'yes':
                return 'useful' 
            else:
                return 'not useful' 
        else:
            return 'not useful' 
    def __call__(self):
        memory = MemorySaver()
        workflow = StateGraph(AgentState)
        workflow.add_node("retriever",self.retrieve)
        workflow.add_node("grade_document",self.grade_document)
        workflow.add_node("generate",self.generate)
        workflow.add_node("transform_query",self.transform_query)

        workflow.add_edge(START,"retriever")
        workflow.add_edge("retriever","grade_document")
        workflow.add_conditional_edges("grade_document",
                                    self.decide_to_generate,
                                    {"transform_query":"transform_query",
                                        "generate":"generate"})
        workflow.add_conditional_edges("generate",
                                    self.grade_generation_vs_documents_and_question,
                                    {"useful":END,
                                        "not useful":'transform_query'})
        workflow.add_conditional_edges("transform_query",
                                    self.decide_to_generate_after_transform,
                                    {"query_not_at_all_relevant":END,
                                        "Retriever":"retriever"})

        self.app = workflow.compile(checkpointer=memory)
        return self.app
if __name__ == "__main__":
    mybot = chatbot()
    app = mybot()
    inputs = {"question": "what is memory in agents?"}
    response=app.invoke(inputs,config=config)["generation"]
    print(type(response))
    print(response)
   
   








