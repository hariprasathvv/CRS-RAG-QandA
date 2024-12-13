from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.callbacks import get_openai_callback
from langchain_community.cache import InMemoryCache
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict, Any
import langchain
import asyncio
import time
import os
from functools import lru_cache

# Enable caching
langchain.cache = InMemoryCache()

# Environment setup
os.environ["OPENAI_API_KEY"] = "KEY"

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str) -> list:
    embeddings = OpenAIEmbeddings()
    return embeddings.embed_query(text)

def create_optimized_prompt():
    system_template = """You are an AI expert with deep expertise in CRS (Cytokine Release Syndrome), CRS drug discovery, CRS biomarkers, and cutting-edge CRS research.

Instructions:
1. Carefully analyze the provided context and synthesize a comprehensive response
2. Draw upon the specific details from the retrieved documents
3. Do not mention the citation numbers.
3. Present information in a clear, organized manner"""

    human_template = """Context from retrieved documents:
{context}

User Question: {question}

Please provide a detailed response that:
- Directly addresses the user's question
- Incorporates specific examples and evidence from the context
- Organizes information in a logical structure"""

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

class OptimizedRetriever:
    def __init__(self, faiss_index):
        self.index = faiss_index
        self.cache = {}
        
    async def get_relevant_documents(self, query: str, k: int = 3):
        """Async retrieval with caching"""
        if query in self.cache:
            return self.cache[query]
        
        embedding = get_cached_embedding(query)
        docs = self.index.similarity_search_by_vector(embedding, k=k)
        self.cache[query] = docs
        return docs

class OptimizedQAChain:
    def __init__(self, faiss_index_path: str):
        self.faiss_index = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        self.retriever = OptimizedRetriever(self.faiss_index)
        self.llm = ChatOpenAI(
            temperature=0.5,
            model_name="gpt-4o-mini",
            max_tokens=4000,
            request_timeout=30,
            streaming=True
        )
        self.prompt = create_optimized_prompt()

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with real-time character-by-character response streaming"""
        metrics = {}
        
        # Measure retrieval time
        retrieval_start = time.time()
        docs = await self.retriever.get_relevant_documents(query)
        metrics['retrieval_time'] = time.time() - retrieval_start

        # Prepare context
        context = "\n".join(doc.page_content for doc in docs)

        # Measure LLM time
        llm_start = time.time()
        messages = self.prompt.format_messages(context=context, question=query)
        response_content = []

        with get_openai_callback() as cb:
            async for chunk in self.llm.astream(messages):
                print(chunk.content, end="", flush=True)  # Stream characters in real-time  # Stream characters in real-time
                response_content.append(chunk.content)
            metrics['llm_time'] = time.time() - llm_start
            metrics['total_tokens'] = cb.total_tokens
            metrics['prompt_tokens'] = cb.prompt_tokens
            metrics['completion_tokens'] = cb.completion_tokens
            metrics['cost'] = cb.total_cost

        return {
            'response': "".join(response_content),
            'metrics': metrics,
            'source_documents': docs
        }


async def main():
    qa_chain = OptimizedQAChain("faiss_index")
    
    print("Optimized CRS Expert System (Type 'exit' to quit)")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break
            
        try:
            # Process query and measure total time
            total_start = time.time()
            result = await qa_chain.process_query(query)
            total_time = time.time() - total_start
            
            # Display performance metrics after response
            print("\n\nPerformance Metrics:")
            metrics = result['metrics']
            print(f"- Retrieval Time: {metrics['retrieval_time']:.2f}s")
            print(f"- LLM Processing Time: {metrics['llm_time']:.2f}s")
            print(f"- Total Response Time: {total_time:.2f}s")
            
            print("\nToken Usage:")
            print(f"- Context Tokens: {metrics['prompt_tokens']}")
            print(f"- Response Tokens: {metrics['completion_tokens']}")
            print(f"- Total Tokens: {metrics['total_tokens']}")
            print(f"- Processing Speed: {metrics['total_tokens']/total_time:.1f} tokens/second")
            
            print(f"\nCost: ${metrics['cost']:.4f}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
