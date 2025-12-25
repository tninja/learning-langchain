import { ChatOpenAI, OpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { PromptTemplate, ChatPromptTemplate } from '@langchain/core/prompts';
import { z } from 'zod';
import { CommaSeparatedListOutputParser } from '@langchain/core/output_parsers';
import { RunnableLambda } from '@langchain/core/runnables';

class TestCh1 {
  async test_a_llm() {
    console.log('\n--- test_a_llm ---');
    const model = new ChatOpenAI({ model: 'gpt-3.5-turbo' });

    const response = await model.invoke('The sky is');
    console.log(response);
  }

  async test_b_chat() {
    console.log('\n--- test_b_chat ---');
    const model = new ChatOpenAI();
    const prompt = [new HumanMessage('What is the capital of France?')];

    const response = await model.invoke(prompt);
    console.log(response);
  }

  async test_c_system() {
    console.log('\n--- test_c_system ---');
    const model = new ChatOpenAI();
    const prompt = [
      new SystemMessage(
        'You are a helpful assistant that responds to questions with three exclamation marks.'
      ),
      new HumanMessage('What is the capital of France?'),
    ];

    const response = await model.invoke(prompt);
    console.log(response);
  }

  async test_d_prompt() {
    console.log('\n--- test_d_prompt ---');
    const template =
      PromptTemplate.fromTemplate(`Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: `);

    const response = await template.invoke({
      context:
        "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
      question: 'Which model providers offer LLMs?',
    });

    console.log(response);
  }

  async test_e_prompt_model() {
    console.log('\n--- test_e_prompt_model ---');
    const model = new OpenAI({
      model: 'gpt-3.5-turbo',
    });
    const template =
      PromptTemplate.fromTemplate(`Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: `);

    const prompt = await template.invoke({
      context:
        "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
      question: 'Which model providers offer LLMs?',
    });

    const response = await model.invoke(prompt);
    console.log(response);
  }

  async test_f_chat_prompt() {
    console.log('\n--- test_f_chat_prompt ---');
    const template = ChatPromptTemplate.fromMessages([
      [
        'system',
        'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
      ],
      ['human', 'Context: {context}'],
      ['human', 'Question: {question}'],
    ]);

    const response = await template.invoke({
      context:
        "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
      question: 'Which model providers offer LLMs?',
    });
    console.log(response);
  }

  async test_g_chat_prompt_model() {
    console.log('\n--- test_g_chat_prompt_model ---');
    const model = new ChatOpenAI();
    const template = ChatPromptTemplate.fromMessages([
      [
        'system',
        'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
      ],
      ['human', 'Context: {context}'],
      ['human', 'Question: {question}'],
    ]);

    const prompt = await template.invoke({
      context:
        "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
      question: 'Which model providers offer LLMs?',
    });

    const response = await model.invoke(prompt);
    console.log(response);
  }

  async test_h_structured() {
    console.log('\n--- test_h_structured ---');
    const answerSchema = z
      .object({
        answer: z.string().describe("The answer to the user's question"),
        justification: z.string().describe('Justification for the answer'),
      })
      .describe(
        "An answer to the user's question along with justification for the answer."
      );

    const model = new ChatOpenAI({
      model: 'gpt-3.5-turbo',
      temperature: 0,
    }).withStructuredOutput(answerSchema);

    const response = await model.invoke(
      'What weighs more, a pound of bricks or a pound of feathers'
    );
    console.log(response);
  }

  async test_i_csv() {
    console.log('\n--- test_i_csv ---');
    const parser = new CommaSeparatedListOutputParser();

    const response = await parser.invoke('apple, banana, cherry');
    console.log(response);
  }

  async test_j_methods() {
    console.log('\n--- test_j_methods ---');
    const model = new ChatOpenAI();

    const response = await model.invoke('Hi there!');
    console.log(response);
    // Hi!

    const completions = await model.batch(['Hi there!', 'Bye!']);
    // ['Hi!', 'See you!']

    for await (const token of await model.stream('Bye!')) {
      console.log(token);
      // Good
      // bye
      // !
    }
  }

  async test_k_imperative() {
    console.log('\n--- test_k_imperative ---');
    // the building blocks

    const template = ChatPromptTemplate.fromMessages([
      ['system', 'You are a helpful assistant.'],
      ['human', '{question}'],
    ]);

    const model = new ChatOpenAI({
      model: 'gpt-3.5-turbo',
    });

    // combine them in a function
    // RunnableLambda adds the same Runnable interface for any function you write

    const chatbot = RunnableLambda.from(async (values: any) => {
      const prompt = await template.invoke(values);
      return await model.invoke(prompt);
    });

    // use it

    const response = await chatbot.invoke({
      question: 'Which model providers offer LLMs?',
    });
    console.log(response);
  }

  async test_ka_stream() {
    console.log('\n--- test_ka_stream ---');
    const template = ChatPromptTemplate.fromMessages([
      ['system', 'You are a helpful assistant.'],
      ['human', '{question}'],
    ]);

    const model = new ChatOpenAI({
      model: 'gpt-3.5-turbo',
    });

    const chatbot = RunnableLambda.from(async function* (values: any) {
      const prompt = await template.invoke(values);
      for await (const token of await model.stream(prompt)) {
        yield token;
      }
    });

    for await (const token of await chatbot.stream({
      question: 'Which model providers offer LLMs?',
    })) {
      console.log(token);
    }
  }

  async test_l_declarative() {
    console.log('\n--- test_l_declarative ---');
    // the building blocks

    const template = ChatPromptTemplate.fromMessages([
      ['system', 'You are a helpful assistant.'],
      ['human', '{question}'],
    ]);

    const model = new ChatOpenAI({
      model: 'gpt-3.5-turbo',
    });

    // combine them in a function

    const chatbot = template.pipe(model);

    // use it

    const response = await chatbot.invoke({
      question: 'Which model providers offer LLMs?',
    });

    console.log(response);

    //streaming

    for await (const part of chatbot.stream({
      question: 'Which model providers offer LLMs?',
    })) {
      console.log(part);
    }
  }
}

async function main() {
  const test = new TestCh1();
  try {
    await test.test_a_llm();
    await test.test_b_chat();
    await test.test_c_system();
    await test.test_d_prompt();
    await test.test_e_prompt_model();
    await test.test_f_chat_prompt();
    await test.test_g_chat_prompt_model();
    await test.test_h_structured();
    await test.test_i_csv();
    await test.test_j_methods();
    await test.test_k_imperative();
    await test.test_ka_stream();
    await test.test_l_declarative();
  } catch (error) {
    console.error('Error running tests:', error);
  }
}

main();
