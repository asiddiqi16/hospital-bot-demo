import dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
reviews_vector_db = Chroma.from_documents(
    reviews, embeddings, persist_directory=REVIEWS_CHROMA_PATH
)

question = """Has anyone complained about
           communication with the hospital staff?"""
relevant_docs = reviews_vector_db.similarity_search(question, k=3)

print(relevant_docs[0].page_content)


print(relevant_docs[1].page_content)


print(relevant_docs[2].page_content)