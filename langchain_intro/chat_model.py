from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import (
    create_tool_calling_agent, # Use create_tool_calling_agent for Gemini
    Tool,
    AgentExecutor,
)
from langchain import hub
# Assuming this is a local tool you have defined
from tools import get_current_wait_time


load_dotenv()

REVIEWS_CHROMA_PATH = "chroma_data/"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=embeddings
)

reviews_retriever  = reviews_vector_db.as_retriever(k=5)



# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params# )





review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)
output_parser = StrOutputParser()
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# review_chain = review_prompt_template | chat_model | output_parser

# context = "I had a great stay!"
# question = "Did anyone have a positive experience?"
# print(review_chain.invoke({"context": context, "question": question}))
review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)
# question = """Has anyone complained about
#         communication with the hospital staff?"""
# print(review_chain.invoke(question))

tools = [
    Tool(
        name="Reviews",
        func=review_chain.invoke, # Use the placeholder or your actual review_chain
        description="""Useful when you need to answer questions
        about patient reviews or experiences at the hospital.
        Not useful for answering questions about specific visit
        details such as payer, billing, treatment, diagnosis,
        chief complaint, hospital, or physician information.
        Pass the entire question as input to the tool. For instance,
        if the question is "What do patients think about the triage system?",
        the input should be "What do patients think about the triage system?"
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_time,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. This tool returns wait times in
        minutes. Do not pass the word "hospital" as input,
        only the hospital name itself. For instance, if the question is
        "What is the wait time at hospital A?", the input should be "A".
        """,
    ),
]

# hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")
# For Gemini, it's generally recommended to use create_tool_calling_agent
# which is a more generic agent constructor that works with any model
# that implements bind_tools. Gemini models inherently support tool calling.

# There is a deprecation warning about this and instructions to move to using LangGraph instead
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

hospital_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),   # <-- For previous messages
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")  # <-- For intermediate steps in agent's reasoning
])


# Create the agent using create_tool_calling_agent for Gemini
# Gemini models leverage the general `bind_tools` functionality.
hospital_agent = create_tool_calling_agent(
    llm=chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

hospital_agent_executor = AgentExecutor(
    agent=hospital_agent,
    tools=tools,
    memory = memory,
    return_intermediate_steps=True,
    verbose=True,
)

hospital_agent_executor.invoke(
    {"input": "What is the current wait time at hospital C?"}
)

hospital_agent_executor.invoke(
    {"input": "What have patients said about their comfort at the hospital?"}
)