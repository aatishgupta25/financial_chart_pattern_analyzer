# explain.py

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple

from model import ChartSenseVLM
from dataset import ChartDataset 

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama 


MODEL_PATH = 'chart_sense_vlm.pt' 
VECTOR_STORE_PATH = 'faiss_index' 
CHART_DIR = 'data/charts' 
LABEL_DIR = 'data/labels' 


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU).")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device (NVIDIA GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU device.")


VLM_MODEL: ChartSenseVLM = None
VOCAB: Dict[str, int] = None
IDX_TO_TOKEN: Dict[int, str] = None
EMBEDDINGS: HuggingFaceEmbeddings = None
RETRIEVER: FAISS = None

vlm_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

def _load_vlm_model() -> Tuple[ChartSenseVLM, Dict[str, int], Dict[int, str]]:
    """The trained VLM model and its associated vocabulary are loaded."""
    global VLM_MODEL, VOCAB, IDX_TO_TOKEN

    if VLM_MODEL is None:
        print(f"Loading VLM model from {MODEL_PATH}...")
        
        temp_dataset = ChartDataset(CHART_DIR, LABEL_DIR, transform=None)
        VOCAB = temp_dataset.vocab
        IDX_TO_TOKEN = temp_dataset.idx_to_token

        model = ChartSenseVLM(vocab=VOCAB)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval() 
        VLM_MODEL = model
        print("VLM model loaded and ready.")
    return VLM_MODEL, VOCAB, IDX_TO_TOKEN

def _load_retriever():
    """Pre-trained embeddings and the FAISS vector store are loaded."""
    global EMBEDDINGS, RETRIEVER
    if RETRIEVER is None:
        print(f"Loading RAG components from {VECTOR_STORE_PATH}...")
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        EMBEDDINGS = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={'device': embeddings_device})
        
        RETRIEVER = FAISS.load_local(VECTOR_STORE_PATH, EMBEDDINGS, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 1})
        print("RAG retriever loaded and ready.")
    return RETRIEVER




def predict_pattern(image_path: str) -> str:
    """
    The candlestick pattern from a given image path is predicted using the VLM.
    This function is designed to be wrapped as a LangChain Tool.
    """
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"

    try:
        model, vocab, idx_to_token = _load_vlm_model()
        
        image = Image.open(image_path).convert("RGB")
        
        image_tensor = vlm_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            
            output_logits = model(image_tensor, max_len=3)
            predicted_token_ids = output_logits.argmax(-1).squeeze(0) 

            decoded_tokens = [idx_to_token[idx.item()] for idx in predicted_token_ids]
            
            pattern = [t for t in decoded_tokens if t not in ['<sos>', '<eos>', '<pad>', '<unk>']]
            
            if len(pattern) > 0:
                return " ".join(pattern)
            else:
                return "no pattern"

    except Exception as e:
        return f"Error predicting pattern: {e}"

def explain_pattern(pattern_name: str) -> str:
    """
    A detailed explanation of a given candlestick pattern is retrieved using the RAG system.
    This function is designed to be wrapped as a LangChain Tool.
    """
    retriever = _load_retriever()
    
    docs = retriever.invoke(pattern_name)
    
    if docs:
        explanation_parts = [doc.page_content for doc in docs]
        return " ".join(explanation_parts)
    else:
        return f"No detailed explanation found for '{pattern_name}' in the knowledge base."



def create_chart_sense_agent():
    try:
        
        llm = ChatOllama(model="gemma3:4b-it-qat", temperature=0.3)
        print("Using Ollama LLM.")
    except Exception as e:
        print(f"Error initializing Ollama LLM. It is ensured that Ollama is running and the specified model is downloaded. Error: {e}")
        print("A fallback to a dummy LLM is being performed, which will limit natural language generation capabilities.")
        from langchain_community.llms import FakeListLLM
        llm = FakeListLLM(responses=["Sorry, I cannot explain without a proper LLM setup."], sequential_responses=True)

    tools = [
        Tool(
            name="chart_to_pattern",
            func=predict_pattern,
            description="Useful for converting a candlestick chart image file path to its predicted pattern name. Input is expected to be the full path to the image file (e.g., 'data/charts/img_00001.png'). The predicted pattern name (e.g., 'doji', 'hammer') or 'no pattern' is returned if no pattern is found. Pattern names are not to be fabricated."
        ),
        Tool(
            name="pattern_explainer",
            func=explain_pattern,
            description="Useful for retrieving a detailed explanation of a given candlestick pattern. Input is expected to be the exact pattern name predicted by chart_to_pattern (e.g., 'doji', 'bullish engulfing'). A textual explanation of the pattern is returned."
        )
    ]

    prompt_template = PromptTemplate.from_template("""
    You are ChartSense++, an AI assistant specializing in candlestick chart analysis and explanations.
    Your objective is to identify candlestick patterns from provided images and subsequently provide detailed explanations of those patterns.

    Access to these tools is provided:

    {tools}

    The following format is to be used:

    Question: the input image path or user query
    Thought: reasoning process is always to be documented
    Action: the action to be taken, must be one of [{tool_names}]
    Action Input: the input for the action
    Observation: the result obtained from the action
    ... (this Thought/Action/Action Input/Observation sequence may repeat as needed)
    Thought: The final answer is now known
    Final Answer: the conclusive answer to the original input question

    Initiation!

    Question: {input}
    {agent_scratchpad}
    """)

    agent = create_react_agent(llm, tools, prompt_template)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

if __name__ == "__main__":
    print("Initializing ChartSense++ Agent...")
    agent_executor = create_chart_sense_agent()
    print("Agent initialized. Chart analysis can now be performed.")
    
    sample_image_path = os.path.join(CHART_DIR, 'img_00146.png')

    if not os.path.exists(sample_image_path):
        print(f"\nWarning: Sample image '{sample_image_path}' was not found.")
        
        image_files = sorted([f for f in os.listdir(CHART_DIR) if f.endswith('.png')])
        if image_files:
            sample_image_path = os.path.join(CHART_DIR, image_files[0])
            print(f"Using first available image: {sample_image_path}")
        else:
            print("No images were found in data/charts. It is recommended to run generate_data.py first.")
            exit()

    print(f"\n Analyzing Chart: {sample_image_path} ")
    
    query = f"Identify the candlestick pattern in the image located at {sample_image_path} and explain its meaning."
    
    try:
        response = agent_executor.invoke({"input": query})
        print("\n  Agent's Final Answer ")
        print(response["output"])
    except Exception as e:
        print(f"\nAn error occurred during agent execution: {e}")
        print("It is ensured that Ollama is running and the specified model is downloaded.")