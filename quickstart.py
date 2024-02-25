from decouple import config
from langchain_openai import ChatOpenAI


openai_key = config("OPENAI_API_KEY", default="")
llm = ChatOpenAI(openai_api_key=openai_key)
# print(llm.invoke("how can langsmith help with testing?").content, end="\n\n")



from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}"),
])

chain = prompt | llm

# print(chain.invoke({"input": "how can langsmith help with testing?"}), end="\n\n")



from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# print(chain.invoke({"input": "What is florazo?"}), end="\n\n")



from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://diccionarioperu.com/significado/florazo")
docs = loader.load()

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

from langchain_core.documents import Document

# document_chain.invoke({
#     'input': 'What is florazo?',
#     'context': [Document(page_content="Florazo significa mentiroso")]
# })


from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "What is florazo?"})
print(response["answer"])

