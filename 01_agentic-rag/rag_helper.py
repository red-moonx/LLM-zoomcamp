# Default system instructions enforcing strict context-grounded answering
INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

# Template combining user query and retrieved context
PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:
    """Base class encapsulating search, prompt construction, LLM calls, and orchestration."""

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='gpt-5.4-mini'
    ):
        # Store injected dependencies and configuration parameters as instance attributes
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        # Search minsearch index applying field boosting and course filtering
        boost_dict = {'question': 3.0, 'section': 0.5}
        filter_dict = {'course': self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        # Format search result dictionaries into a clean, structured text block
        lines = []

        for doc in search_results:
            lines.append(doc['section'])
            lines.append('Q: ' + doc['question'])
            lines.append('A: ' + doc['answer'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        # Combine retrieved context and user query into the prompt template
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        # Send developer instructions and prompt to OpenAI using responses.create API
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

    def rag(self, query):
        # Orchestrate complete RAG pipeline: search -> build prompt -> generate response
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer

