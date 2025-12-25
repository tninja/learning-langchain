import unittest
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain.indexes import SQLRecordManager, index
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import uuid
from ragatouille import RAGPretrainedModel
import requests

class TestCh2(unittest.TestCase):

    def test_a_text_loader(self):
        loader = TextLoader('./test.txt', encoding="utf-8")
        docs = loader.load()
        print(docs)

    def test_b_web_loader(self):
        loader = WebBaseLoader('https://www.langchain.com/')
        docs = loader.load()
        print(docs)

    def test_c_pdf_loader(self):
        loader = PyPDFLoader('./test.pdf')
        pages = loader.load()
        print(pages)

    def test_d_rec_text_splitter(self):
        loader = TextLoader('./test.txt', encoding="utf-8")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_docs = splitter.split_documents(docs)

        print(splitted_docs)

    def test_e_rec_text_splitter_code(self):
        PYTHON_CODE = """ def hello_world(): print(\"Hello, World!\") # Call the function hello_world() """

        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=50, chunk_overlap=0
        )

        python_docs = python_splitter.create_documents([PYTHON_CODE])

        print(python_docs)

    def test_f_markdown_splitter(self):
        markdown_text = """ # 🦜🔗 LangChain ⚡ Building applications with LLMs through composability ⚡ ## Quick Install ```bash pip install langchain ``` As an open source project in a rapidly developing field, we are extremely open     to contributions. """

        md_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
        )

        md_docs = md_splitter.create_documents(
            [markdown_text], [{"source": "https://www.langchain.com"}])

        print(md_docs)

    def test_g_embeddings(self):
        model = OpenAIEmbeddings(model="text-embedding-3-small")
        embeddings = model.embed_documents([
            "Hi there!",
            "Oh, hello!",
            "What's your name?",
            "My friends call me World",
            "Hello World!"
        ])

        print(embeddings)

    def test_h_load_split_embed(self):
        # Load the document
        loader = TextLoader("./test.txt", encoding="utf-8")
        doc = loader.load()

        # Split the document
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(doc)

        # Generate embeddings
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        embeddings = embeddings_model.embed_documents(
            [chunk.page_content for chunk in chunks]
        )

        print(embeddings)

    def test_i_pg_vector(self):
        # See docker command above to launch a postgres instance with pgvector enabled.
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

        # Load the document, split it into chunks
        raw_documents = TextLoader('./test.txt', encoding="utf-8").load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)

        # Create embeddings for the documents
        embeddings_model = OpenAIEmbeddings()

        db = PGVector.from_documents(
            documents, embeddings_model, connection=connection)

        results = db.similarity_search("query", k=4)

        print(results)

        print("Adding documents to the vector store")
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        db.add_documents(
            [
                Document(
                    page_content="there are cats in the pond",
                    metadata={"location": "pond", "topic": "animals"},
                ),
                Document(
                    page_content="ducks are also found in the pond",
                    metadata={"location": "pond", "topic": "animals"},
                ),
            ],
            ids=ids,
        )

        print("Documents added successfully.\n Fetched documents count:",
              len(db.get_by_ids(ids)))

        print("Deleting document with id", ids[1])
        db.delete({"ids": ids})

        print("Document deleted successfully.\n Fetched documents count:",
              len(db.get_by_ids(ids)))

    def test_j_record_manager(self):
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        collection_name = "my_docs"
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        namespace = "my_docs_namespace"

        vectorstore = PGVector(
            embeddings=embeddings_model,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )

        record_manager = SQLRecordManager(
            namespace,
            db_url="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
        )

        # Create the schema if it doesn't exist
        record_manager.create_schema()

        # Create documents
        docs = [
            Document(page_content='there are cats in the pond', metadata={
                     "id": 1, "source": "cats.txt"}),
            Document(page_content='ducks are also found in the pond', metadata={
                     "id": 2, "source": "ducks.txt"}),
        ]

        # Index the documents
        index_1 = index(
            docs,
            record_manager,
            vectorstore,
            cleanup="incremental",  # prevent duplicate documents
            source_id_key="source",  # use the source field as the source_id
        )

        print("Index attempt 1:", index_1)

        # second time you attempt to index, it will not add the documents again
        index_2 = index(
            docs,
            record_manager,
            vectorstore,
            cleanup="incremental",
            source_id_key="source",
        )

        print("Index attempt 2:", index_2)

        # If we mutate a document, the new version will be written and all old versions sharing the same source will be deleted.

        docs[0].page_content = "I just modified this document!"

        index_3 = index(
            docs,
            record_manager,
            vectorstore,
            cleanup="incremental",
            source_id_key="source",
        )

        print("Index attempt 3:", index_3)

    def test_k_multi_vector_retriever(self):
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        collection_name = "summaries"
        embeddings_model = OpenAIEmbeddings()
        # Load the document
        loader = TextLoader("./test.txt", encoding="utf-8")
        docs = loader.load()

        print("length of loaded docs: ", len(docs[0].page_content))
        # Split the document
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # The rest of your code remains the same, starting from:
        prompt_text = "Summarize the following document:\n\n{doc}"

        prompt = ChatPromptTemplate.from_template(prompt_text)
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        summarize_chain = { "doc": lambda x: x.page_content} | prompt | llm | StrOutputParser()

        # batch the chain across the chunks
        summaries = summarize_chain.batch(chunks, {"max_concurrency": 5})

        # The vectorstore to use to index the child chunks
        vectorstore = PGVector(
            embeddings=embeddings_model,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
        # The storage layer for the parent documents
        store = InMemoryStore()
        id_key = "doc_id"

        # indexing the summaries in our vector store, whilst retaining the original documents in our document store:
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        # Changed from summaries to chunks since we need same length as docs
        doc_ids = [str(uuid.uuid4()) for _ in chunks]

        # Each summary is linked to the original document by the doc_id
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]

        # Add the document summaries to the vector store for similarity search
        retriever.vectorstore.add_documents(summary_docs)

        # Store the original documents in the document store, linked to their summaries via doc_ids
        # This allows us to first search summaries efficiently, then fetch the full docs when needed
        retriever.docstore.mset(list(zip(doc_ids, chunks)))

        # vector store retrieves the summaries
        sub_docs = retriever.vectorstore.similarity_search(
            "chapter on philosophy", k=2)

        print("sub docs: ", sub_docs[0].page_content)

        print("length of sub docs:\n", len(sub_docs[0].page_content))

        # Whereas the retriever will return the larger source document chunks:
        retrieved_docs = retriever.invoke("chapter on philosophy")

        print("length of retrieved docs: ", len(retrieved_docs[0].page_content))

    def test_l_rag_colbert(self):
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

        def get_wikipedia_page(title: str):
            """
            Retrieve the full text content of a Wikipedia page.
            :param title: str - Title of the Wikipedia page.
            :return: str - Full text content of the page as raw string.
            """
            # Wikipedia API endpoint
            URL = "https://en.wikipedia.org/w/api.php"
            # Parameters for the API request
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            }
            # Custom User-Agent header to comply with Wikipedia's best practices
            headers = {"User-Agent": "RAGatouille_tutorial/0.0.1"}
            response = requests.get(URL, params=params, headers=headers)
            data = response.json()
            # Extracting page content
            page = next(iter(data["query"]["pages"].values()))
            return page["extract"] if "extract" in page else None

        full_document = get_wikipedia_page("Hayao_Miyazaki")
        # Create an index
        RAG.index(
            collection=[full_document],
            index_name="Miyazaki-123",
            max_document_length=180,
            split_documents=True,
        )
        # query
        results = RAG.search(query="What animation studio did Miyazaki found?", k=3)

        print(results)

        # Alternative: Utilize langchain retriever
        retriever = RAG.as_langchain_retriever(k=3)
        retriever.invoke("What animation studio did Miyazaki found?")

if __name__ == '__main__':
    unittest.main()
