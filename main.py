import os
import torch
from huggingface_hub import login
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.graph import KnowledgeGraph, Node, NodeType 
from ragas.testset.synthesizers import default_query_distribution


# Initialize CUDA if available
torch.cuda.init()

# Set your OpenAI API Key
#os.environ["OPENAI_API_KEY"] = None

# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

# Configure CUDA environment variables
os.environ["CUDA_DEVICE_ORDER"]="00000000:2F:00.0"# "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # specify which GPU(s) to be used
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"]="1"


# Load documents
path="/home/jovyan/Documentos"
print("Loading documents:")
loader = DirectoryLoader(path, show_progress=True)
docs = loader.load()
print("Documents loaded successfully.")

'''
print("is cuda available?",torch.cuda.is_available())
device = torch.cuda.current_device() if torch.cuda.is_available() else -1
print("device:",device)
'''
# Initialize the Hugging Face pipeline
huggingface_llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.1-8B",
    task="text-generation",
    device=-1, # -1 for CPU, 0 for GPU
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=True,
        repetition_penalty=1.03,
        temperature=0.1,
    ),
)

print("Model loaded successfully.")
print("Tipo de huggingface_llm:", type(huggingface_llm))


'''
# Test the model
prompt = "Once upon a time"
generated_text = huggingface_llm.invoke(prompt)
print("***********************\n")
print("Generated text:", generated_text)
print("***********************\n")
'''
#model_id = "meta-llama/Llama-3.1-8B"
#tokenizer = AutoTokenizer.from_pretrained(model_id)
#huggingface_llm = ChatHuggingFace(llm=huggingface_llm, tokenizer=tokenizer, verbose=True)
#print("Tipo de huggingface_llm al aplicar el chat:", type(huggingface_llm))

# Initialize the embeddings model
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}, #{'device': 'cuda'} if cuda available
    encode_kwargs={'normalize_embeddings': False, "show_progress_bar": True},
    multi_process=True,
    show_progress=True
)
print("Tipo de huggingface_embeddings:", type(huggingface_embeddings))
print("Embeddings configured successfully.")


# Generate testset
generator_llm = LangchainLLMWrapper(huggingface_llm)
generator_embeddings = LangchainEmbeddingsWrapper(huggingface_embeddings)
#generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
#generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
print("Tipo de generator_llm:", type(generator_llm))
print("Tipo de generator_embeddings:", type(generator_embeddings))

#generator = TestsetGenerator.from_langchain(llm=generator_llm, embedding_model=generator_embeddings)
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
print("Generator initialized.")
print("Tipo de generator:", type(generator))

dataset = generator.generate_with_langchain_docs(docs, testset_size=5, with_debugging_logs=True, raise_exceptions=False)
print("Dataset generated.")

# Exporting and analyzing results
df_dataset = dataset.to_pandas()
df_dataset.to_csv('dataset.csv', index=False)
print("Generaded testset and save in 'dataset.csv'.")
