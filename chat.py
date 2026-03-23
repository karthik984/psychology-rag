import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv
from utils import get_embedding

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "psychology"
TOP_K = 5

client = Anthropic()
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def retrieve(question: str) -> list[str]:
    embedding = get_embedding([question])
    results = collection.query(
        query_embeddings=embedding,
        n_results=TOP_K
    )
    docs = results["documents"]
    if docs is None:
        return []
    return docs[0]

def ask(question: str, conversation_history: list) -> str:
    #Retrieve relevant chunks
    chunks = retrieve(question)
    context = "\n\n---\n\n".join(chunks)

    # Build the user message with context injected
    user_message = f"""Use the following excerpts from a psychology textbook to answer the question.
        If the answer isn't in the excerpts, say so and DON'T MAKE THINGS UP!

        CONTEXT:
        {context}

        QUESTION:
        {question}"""

    # Append to conversation history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a helpful assistant that answers questions about psychology based strictly on the provided context.",
        messages=conversation_history
    )

    answer = response.content[0].text # type: ignore

    # Append Claude's response to history
    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer

def main():
    print("Psychology RAG Chatbot")
    print("Type 'quit' to exit, 'clear' to reset conversation history\n")

    conversation_history = []

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() == "quit":
            print("Bye!")
            break
        if question.lower() == "clear":
            conversation_history = []
            print("Conversation History Cleared.\n")
            continue

        answer = ask(question, conversation_history)
        print(f"\nAssitance:{answer}")

if __name__ == "__main__":
    main()