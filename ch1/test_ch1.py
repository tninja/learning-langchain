import unittest
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from pydantic import BaseModel

class TestCh1(unittest.IsolatedAsyncioTestCase):

    def test_a_llm(self):
        model = ChatOpenAI(model="gpt-4.1-mini")
        response = model.invoke("The sky is")
        print(response.content)

    def test_b_chat(self):
        model = ChatOpenAI()
        prompt = [HumanMessage("What is the capital of France?")]
        response = model.invoke(prompt)
        print(response.content)

    def test_c_system(self):
        model = ChatOpenAI()
        system_msg = SystemMessage(
            "You are a helpful assistant that responds to questions with three exclamation marks."
        )
        human_msg = HumanMessage("What is the capital of France?")
        response = model.invoke([system_msg, human_msg])
        print(response.content)

    def test_d_prompt(self):
        template = PromptTemplate.from_template("""Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)
        response = template.invoke(
            {
                "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
                "question": "Which model providers offer LLMs?",
            }
        )
        print(response)

    def test_e_prompt_model(self):
        # both `template` and `model` can be reused many times
        template = PromptTemplate.from_template("""Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)

        model = ChatOpenAI(model="gpt-4.1-mini")

        # `prompt` and `completion` are the results of using template and model once
        prompt = template.invoke(
            {
                "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
                "question": "Which model providers offer LLMs?",
            }
        )

        response = model.invoke(prompt)
        print(response)

    def test_f_chat_prompt(self):
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
                ),
                ("human", "Context: {context}"),
                ("human", "Question: {question}"),
            ]
        )

        response = template.invoke(
            {
                "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
                "question": "Which model providers offer LLMs?",
            }
        )
        print(response)

    def test_g_chat_prompt_model(self):
        # both `template` and `model` can be reused many times
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
                ),
                ("human", "Context: {context}"),
                ("human", "Question: {question}"),
            ]
        )

        model = ChatOpenAI()

        # `prompt` and `completion` are the results of using template and model once
        prompt = template.invoke(
            {
                "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
                "question": "Which model providers offer LLMs?",
            }
        )

        print(model.invoke(prompt))

    def test_h_structured(self):
        class AnswerWithJustification(BaseModel):
            """An answer to the user's question along with justification for the answer."""

            answer: str
            """The answer to the user's question"""
            justification: str
            """Justification for the answer"""

        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        structured_llm = llm.with_structured_output(AnswerWithJustification)

        response = structured_llm.invoke(
            "What weighs more, a pound of bricks or a pound of feathers")
        print(response)

    def test_i_csv(self):
        parser = CommaSeparatedListOutputParser()
        response = parser.invoke("apple, banana, cherry")
        print(response)

    def test_j_methods(self):
        model = ChatOpenAI(model="gpt-4.1-mini")

        completion = model.invoke("Hi there!")
        # Hi!

        completions = model.batch(["Hi there!", "Bye!"])
        # ['Hi!', 'See you!']

        for token in model.stream("Bye!"):
            print(token)
            # Good
            # bye
            # !

    def test_k_imperative(self):
        # the building blocks
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{question}"),
            ]
        )

        model = ChatOpenAI(model="gpt-4.1-mini")

        # combine them in a function
        # @chain decorator adds the same Runnable interface for any function you write
        @chain
        def chatbot(values):
            prompt = template.invoke(values)
            return model.invoke(prompt)

        # use it
        response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
        print(response.content)

    def test_ka_stream(self):
        model = ChatOpenAI(model="gpt-4.1-mini")

        template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{question}"),
            ]
        )

        @chain
        def chatbot(values):
            prompt = template.invoke(values)
            for token in model.stream(prompt):
                yield token

        for part in chatbot.stream({"question": "Which model providers offer LLMs?"}):
            print(part)

    async def test_kb_async(self):
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{question}"),
            ]
        )

        model = ChatOpenAI(model="gpt-4.1-mini")

        @chain
        async def chatbot(values):
            prompt = await template.ainvoke(values)
            return await model.ainvoke(prompt)

        response = await chatbot.ainvoke({"question": "Which model providers offer LLMs?"})
        print(response)

    def test_l_declarative(self):
        # the building blocks
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{question}"),
            ]
        )

        model = ChatOpenAI()

        # combine them with the | operator
        chatbot = template | model

        # use it
        response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
        print(response.content)

        # streaming
        for part in chatbot.stream({"question": "Which model providers offer LLMs?"}):
            print(part)

if __name__ == '__main__':
    unittest.main()
