import unittest
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableLambda

class TestCh3(unittest.TestCase):

    def test_a_basic_rag(self):
        # See docker command above to launch a postgres instance with pgvector enabled.
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

        # Load the document, split it into chunks
        raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)

        # Create embeddings for the documents
        embeddings_model = OpenAIEmbeddings()

        db = PGVector.from_documents(
            documents, embeddings_model, connection=connection)

        # create retriever to retrieve 2 relevant documents
        retriever = db.as_retriever(search_kwargs={"k": 2})

        query = 'Who are the key figures in the ancient greek history of philosophy?'

        # fetch relevant documents
        docs = retriever.invoke(query)

        print(docs[0].page_content)

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context: {context} Question: {question} """
        )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        llm_chain = prompt | llm

        # answer the question based on relevant documents
        result = llm_chain.invoke({"context": docs, "question": query})

        print(result)
        print("\n\n")

        # Run again but this time encapsulate the logic for efficiency

        # @chain decorator transforms this function into a LangChain runnable,
        # making it compatible with LangChain's chain operations and pipeline

        print("Running again but this time encapsulate the logic for efficiency\n")

        @chain
        def qa(input):
            # fetch relevant documents
            docs = retriever.invoke(input)
            # format prompt
            formatted = prompt.invoke({"context": docs, "question": input})
            # generate answer
            answer = llm.invoke(formatted)
            return answer

        # run it
        result = qa.invoke(query)
        print(result.content)

    def test_b_rewrite(self):
        # See docker command above to launch a postgres instance with pgvector enabled.
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

        # Load the document, split it into chunks
        raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)

        # Create embeddings for the documents
        embeddings_model = OpenAIEmbeddings()

        db = PGVector.from_documents(
            documents, embeddings_model, connection=connection)

        # create retriever to retrieve 2 relevant documents
        retriever = db.as_retriever(search_kwargs={"k": 2})

        # Query starts with irrelevant information before asking the relevant question
        query = 'Today I woke up and brushed my teeth, then I sat down to read the news. But then I forgot the food on the cooker. Who are some key figures in the ancient greek history of philosophy?'

        # fetch relevant documents
        docs = retriever.invoke(query)

        print(docs[0].page_content)
        print("\n\n")

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context: {context} Question: {question} """
        )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Run again but this time encapsulate the logic for efficiency

        # @chain decorator transforms this function into a LangChain runnable,
        # making it compatible with LangChain's chain operations and pipeline

        @chain
        def qa(input):
            # fetch relevant documents
            docs = retriever.invoke(input)
            # format prompt
            formatted = prompt.invoke({"context": docs, "question": input})
            # generate answer
            answer = llm.invoke(formatted)
            return answer

        # run it
        result = qa.invoke(query)
        print(result.content)

        print("\nRewrite the query to improve accuracy\n")

        rewrite_prompt = ChatPromptTemplate.from_template(
            """Provide a better search query for web search engine to answer the given question, end the queries with ’**’. Question: {x} Answer:"""
        )

        def parse_rewriter_output(message):
            return message.content.strip('"').strip("**")

        rewriter = rewrite_prompt | llm | parse_rewriter_output

        @chain
        def qa_rrr(input):
            # rewrite the query
            new_query = rewriter.invoke(input)
            print("Rewritten query: ", new_query)
            # fetch relevant documents
            docs = retriever.invoke(new_query)
            # format prompt
            formatted = prompt.invoke({"context": docs, "question": input})
            # generate answer
            answer = llm.invoke(formatted)
            return answer

        print("\nCall model again with rewritten query\n")

        # call model again with rewritten query
        result = qa_rrr.invoke(query)
        print(result.content)

    def test_c_multi_query(self):
        # See docker command above to launch a postgres instance with pgvector enabled.
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

        # Load the document, split it into chunks
        raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)

        # Create embeddings for the documents
        embeddings_model = OpenAIEmbeddings()

        db = PGVector.from_documents(
            documents, embeddings_model, connection=connection)

        # create retriever to retrieve 2 relevant documents
        retriever = db.as_retriever(search_kwargs={"k": 5})

        # instruction to generate multiple queries
        perspectives_prompt = ChatPromptTemplate.from_template(
            """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. 
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based  similarity search. 
            Provide these alternative questions separated by newlines. 
            Original question: {question}"""
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo")

        def parse_queries_output(message):
            return message.content.split('\n')

        query_gen = perspectives_prompt | llm | parse_queries_output

        def get_unique_union(document_lists):
            # Flatten list of lists, and dedupe them
            deduped_docs = {doc.page_content: doc for sublist in document_lists for doc in sublist}
            # return a flat list of unique docs
            return list(deduped_docs.values())

        retrieval_chain = query_gen | retriever.batch | get_unique_union

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context: {context} Question: {question} """
        )

        query = "Who are the key figures in the ancient greek history of philosophy?"

        @chain
        def multi_query_qa(input):
            # fetch relevant documents
            docs = retrieval_chain.invoke(input)  # format prompt
            formatted = prompt.invoke(
                {"context": docs, "question": input})  # generate answer
            answer = llm.invoke(formatted)
            return answer

        # run
        print("Running multi query qa\n")
        result = multi_query_qa.invoke(query)
        print(result.content)

    def test_d_rag_fusion(self):
        # See docker command above to launch a postgres instance with pgvector enabled.
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

        # Load the document, split it into chunks
        raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)

        # Create embeddings for the documents
        embeddings_model = OpenAIEmbeddings()

        db = PGVector.from_documents(
            documents, embeddings_model, connection=connection)

        # create retriever to retrieve 2 relevant documents
        retriever = db.as_retriever(search_kwargs={"k": 5})

        prompt_rag_fusion = ChatPromptTemplate.from_template(
            """You are a helpful assistant that generates multiple search queries based on a single input query. \n Generate multiple search queries related to: {question} \n Output (4 queries):"""
        )

        def parse_queries_output(message):
            return message.content.split('\n')

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        query_gen = prompt_rag_fusion | llm | parse_queries_output

        query = "Who are the key figures in the ancient greek history of philosophy?"

        generated_queries = query_gen.invoke(query)

        print("generated queries: ", generated_queries)

        """
        we fetch relevant documents for each query and pass them into a function to rerank (that is, reorder according to relevancy) the final list of relevant documents.
        """

        def reciprocal_rank_fusion(results: list[list], k=60):
            """reciprocal rank fusion on multiple lists of ranked documents and an optional parameter k used in the RRF formula"""
            # Initialize a dictionary to hold fused scores for each document
            # Documents will be keyed by their contents to ensure uniqueness
            fused_scores = {}
            documents = {}
            for docs in results:
                # Iterate through each document in the list, with its rank (position in the list)
                for rank, doc in enumerate(docs):
                    doc_str = doc.page_content
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                        documents[doc_str] = doc
                    fused_scores[doc_str] += 1 / (rank + k)
            # sort the documents based on their fused scores in descending order to get the final reranked results
            reranked_doc_strs = sorted(
                fused_scores, key=lambda d: fused_scores[d], reverse=True)
            return [documents[doc_str] for doc_str in reranked_doc_strs]

        retrieval_chain = query_gen | retriever.batch | reciprocal_rank_fusion

        result = retrieval_chain.invoke(query)

        print("retrieved context using rank fusion: ", result[0].page_content)
        print("\n\n")

        print("Use model to answer question based on retrieved docs\n")

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context: {context} Question: {question} """
        )

        query = "Who are the some important yet not well known philosophers in the ancient greek history of philosophy?"

        @chain
        def rag_fusion(input):
            # fetch relevant documents
            docs = retrieval_chain.invoke(input)  # format prompt
            formatted = prompt.invoke(
                {"context": docs, "question": input})  # generate answer
            answer = llm.invoke(formatted)
            return answer

        # run
        print("Running rag fusion\n")
        result = rag_fusion.invoke(query)
        print(result.content)

    def test_e_hyde(self):
        # See docker command above to launch a postgres instance with pgvector enabled.
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

        # Load the document, split it into chunks
        raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)

        # Create embeddings for the documents
        embeddings_model = OpenAIEmbeddings()

        db = PGVector.from_documents(
            documents, embeddings_model, connection=connection)

        # create retriever to retrieve 2 relevant documents
        retriever = db.as_retriever(search_kwargs={"k": 5})

        prompt_hyde = ChatPromptTemplate.from_template(
            """Please write a passage to answer the question.\n Question: {question} \n Passage:"""
        )

        generate_doc = (prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser())

        """
        Next, we take the hypothetical document generated above and use it as input to the retriever, 
        which will generate its embedding and search for similar documents in the vector store:
        """
        retrieval_chain = generate_doc | retriever

        query = "Who are some lesser known philosophers in the ancient greek history of philosophy?"

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context: {context} Question: {question} """
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        @chain
        def qa(input):
            # fetch relevant documents from the hyde retrieval chain defined earlier
            docs = retrieval_chain.invoke(input)
            # format prompt
            formatted = prompt.invoke({"context": docs, "question": input})
            # generate answer
            answer = llm.invoke(formatted)
            return answer

        print("Running hyde\n")
        result = qa.invoke(query)
        print("\n\n")
        print(result.content)

    def test_f_router(self):
        # Data model class
        class RouteQuery(BaseModel):
            """Route a user query to the most relevant datasource."""
            datasource: Literal["python_docs", "js_docs"] = Field(
                ...,
                description="Given a user question, choose which datasource would be most relevant for answering their question",
            )

        # Prompt template
        # LLM with function call
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        """
        with_structured_output: Model wrapper that returns outputs formatted to match the given schema.

        """
        structured_llm = llm.with_structured_output(RouteQuery)

        # Prompt
        system = """You are an expert at routing a user question to the appropriate data source. Based on the programming language the question is referring to, route it to the relevant data source."""
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", "{question}")]
        )

        # Define router
        router = prompt | structured_llm

        # Run
        question = """Why doesn't the following code work: 
        from langchain_core.prompts 
        import ChatPromptTemplate 
        prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"]) 
        prompt.invoke("french") """

        result = router.invoke({"question": question})
        print("\nRouting to: ", result)

        """
        Once we extracted the relevant data source, we can pass the value into another function to execute additional logic as required:
        """

        def choose_route(result):
            if "python_docs" in result.datasource.lower():
                return "chain for python_docs"
            else:
                return "chain for js_docs"

        full_chain = router | RunnableLambda(choose_route)

        result = full_chain.invoke({"question": question})
        print("\nChoose route: ", result)

    def test_g_semantic_router(self):
        physics_template = """You are a very smart physics professor. You are great at     answering questions about physics in a concise and easy-to-understand manner.     When you don't know the answer to a question, you admit that you don't know. Here is a question: {query}"""
        math_template = """You are a very good mathematician. You are great at answering     math questions. You are so good because you are able to break down hard     problems into their component parts, answer the component parts, and then     put them together to answer the broader question. Here is a question: {query}"""

        # Embed prompts
        embeddings = OpenAIEmbeddings()
        prompt_templates = [physics_template, math_template]
        prompt_embeddings = embeddings.embed_documents(prompt_templates)

        # Route question to prompt

        @chain
        def prompt_router(query):
            query_embedding = embeddings.embed_query(query)
            similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
            most_similar = prompt_templates[similarity.argmax()]
            print("Using MATH" if most_similar == math_template else "Using PHYSICS")
            return PromptTemplate.from_template(most_similar)

        semantic_router = (prompt_router | ChatOpenAI() | StrOutputParser())

        result = semantic_router.invoke("What's a black hole")
        print("\nSemantic router result: ", result)

    def test_h_self_query(self):
        # See docker command above to launch a postgres instance with pgvector enabled.
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

        docs = [
            Document(
                page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
                metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
            ),
            Document(
                page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
                metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
            ),
            Document(
                page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
                metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
            ),
            Document(
                page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
                metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
            ),
            Document(
                page_content="Toys come alive and have a blast doing so",
                metadata={"year": 1995, "genre": "animated"},
            ),
            Document(
                page_content="Three men walk into the Zone, three men walk out of the Zone",
                metadata={
                    "year": 1979,
                    "director": "Andrei Tarkovsky",
                    "genre": "thriller",
                    "rating": 9.9,
                },
            ),
        ]

        # Create embeddings for the documents
        embeddings_model = OpenAIEmbeddings()

        vectorstore = PGVector.from_documents(
            docs, embeddings_model, connection=connection)

        # Define the fields for the query
        fields = [
            AttributeInfo(
                name="genre",
                description="The genre of the movie",
                type="string or list[string]",
            ),
            AttributeInfo(
                name="year",
                description="The year the movie was released",
                type="integer",
            ),
            AttributeInfo(
                name="director",
                description="The name of the movie director",
                type="string",
            ),
            AttributeInfo(
                name="rating",
                description="A 1-10 rating for the movie",
                type="float",
            ),
        ]

        description = "Brief summary of a movie"
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        retriever = SelfQueryRetriever.from_llm(llm, vectorstore, description, fields)

        # This example only specifies a filter
        print(retriever.invoke("I want to watch a movie rated higher than 8.5"))

        print('\n')

        # This example specifies multiple filters
        print(retriever.invoke(
            "What's a highly rated (above 8.5) science fiction film?"))

    def test_i_sql_example(self):
        db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        print(db.get_usable_table_names())
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # convert question to sql query
        write_query = create_sql_query_chain(llm, db)

        # Execute SQL query
        execute_query = QuerySQLDatabaseTool(db=db)

        # combined chain = write_query | execute_query
        combined_chain = write_query | execute_query

        # run the chain
        result = combined_chain.invoke({"question": "How many employees are there?"})

        print(result)
if __name__ == '__main__':
    unittest.main()
