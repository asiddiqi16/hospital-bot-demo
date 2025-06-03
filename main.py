import os

from dotenv import load_dotenv
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from google import genai

load_dotenv()

# Get an environment variable
# GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
os.environ.get("GOOGLE_API_KEY")


def main():
    print("Hello from hospital-bot-demo!")

    # client = genai.Client(api_key=GEMINI_API_KEY)

    # response = client.models.generate_content(
    #     model="gemini-2.0-flash", contents="Explain how AI works in a few words"
    # )
    # print(response.text)


    model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    # print(model.invoke("Explain how AI works in a few words"))
    messages = [
    SystemMessage(
        content="""You're an assistant knowledgeable about
        healthcare. Only answer healthcare-related questions."""
    ),
    HumanMessage(content="What is Medicaid managed care?"),
]
    # print(model.invoke(messages))

    review_template_str = """Your job is to use patient
    reviews to answer questions about their experience at a hospital.
    Use the following context to answer questions. Be as detailed
    as possible, but don't make up any information that's not
    from the context. If you don't know an answer, say you don't know.
    {context}
    {question}
    """
    review_template = ChatPromptTemplate.from_template(review_template_str)
    context = "I had a great stay!"
    question = "Did anyone have a positive experience?"
    review_template.format(context=context, question=question)

    

if __name__ == "__main__":
    main()
