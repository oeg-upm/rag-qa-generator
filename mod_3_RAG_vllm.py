"""
Módulo RAG: implementación de un sistema de Recuperación Aumentada de Generación (RAG)
usando vLLM para el modelo de lenguaje y Ollama para embeddings.
"""
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from openai import OpenAI

# Cargador de PDFs
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# Esquema de documentos de LangChain
from langchain.schema import Document

# Herramienta para dividir textos en chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store FAISS
from langchain_community.vectorstores import FAISS

# Integración de Ollama para embeddings
from langchain_ollama import OllamaEmbeddings

# Plantillas de prompt
from langchain.prompts import PromptTemplate


def load_documents_from_dir(dir_path: str) -> list[Document]:
    """
    Recorre recursivamente dir_path buscando archivos .txt, .pdf y .docx
    y devuelve una lista de Document con su contenido.
    """
    docs: list[Document] = []
    base_path = Path(dir_path)
    
    if not base_path.exists():
        print(f"⚠️ El directorio {dir_path} no existe")
        return docs
    
    # Buscar archivos de diferentes tipos
    file_patterns = ["*.txt", "*.pdf", "*.docx"]
    
    for pattern in file_patterns:
        files = list(base_path.rglob(pattern))
        
        for file_path in files:
            try:
                # Seleccionar el loader apropiado según la extensión
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                elif file_path.suffix.lower() == '.txt':
                    loader = TextLoader(str(file_path), encoding='utf-8')
                elif file_path.suffix.lower() == '.docx':
                    loader = Docx2txtLoader(str(file_path))
                else:
                    continue
                
                loaded = loader.load()
                docs.extend(loaded)
                
            except Exception as e:
                print(f"⚠️ Error cargando {file_path}: {e}")
    
    return docs


def generar_respuestas_rag(csv_path, documents_dir, output_path, 
                          embedding_model="jina/jina-embeddings-v2-base-es:latest", 
                          llm_model="NousResearch/Meta-Llama-3-8B-Instruct",
                          vllm_base_url="http://localhost:8000/v1/",
                          vllm_api_key="not-needed"):
    """
    Función principal para generar respuestas RAG desde un CSV usando vLLM.
    
    Args:
        csv_path: Ruta al CSV con las preguntas
        documents_dir: Directorio con los PDFs para indexar
        output_path: Ruta donde guardar el CSV resultante
        embedding_model: Modelo de embeddings para RAG (Ollama)
        llm_model: Modelo LLM para RAG (vLLM)
        vllm_base_url: URL base del servidor vLLM
        vllm_api_key: Clave API para vLLM
    """
    print(f"📊 Cargando dataset desde: {csv_path}")
    
    # 1) Leemos el CSV
    df = pd.read_csv(csv_path)
    
    # Validar que existan las columnas esperadas
    required = ["input"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Falta la columna requerida en el CSV: '{col}'")
    
    # Verificar si existe columna reformulada
    has_reformulated = "input_reformulado_2" in df.columns
    if has_reformulated:
        print("✅ Encontrada columna 'input_reformulado_2', se procesarán ambas versiones.")
    else:
        print("⚠️  No se encontró 'input_reformulado_2', solo se procesará 'input'.")
    
    # 2) Cargamos los documentos desde la carpeta
    print(f"📂 Cargando documentos desde: {documents_dir}")
    docs = load_documents_from_dir(documents_dir)

    # Verificar que se cargaron documentos
    if len(docs) == 0:
        raise ValueError(f"❌ No se encontraron documentos válidos en {documents_dir}. "
                        f"Asegúrate de que el directorio contenga archivos .txt, .pdf o .docx")
    
    print(f"✅ {len(docs)} documentos cargados.")
    
    # 3) Inicializamos el RAG con los documentos
    print(f"🔧 Inicializando RAG con modelo de embeddings: {embedding_model}")
    print(f"🔧 Modelo LLM (vLLM): {llm_model}")
    
    rag = RAG(
        docs, 
        embedding_model=embedding_model,
        llm_model=llm_model,
        vllm_base_url=vllm_base_url,
        vllm_api_key=vllm_api_key
    )
    
    # 4) Preparamos las columnas de salida si no existen
    if 'actual_output' not in df.columns:
        df['actual_output'] = ''
    if 'retrieval_context' not in df.columns:
        df['retrieval_context'] = ''
        
    if has_reformulated:
        if 'actual_output_reformulado' not in df.columns:
            df['actual_output_reformulado'] = ''
        if 'retrieval_context_reformulado' not in df.columns:
            df['retrieval_context_reformulado'] = ''
    
    # Función para procesar de forma segura CON contexto
    def process_safely_with_context(question):
        try:
            if pd.isna(question) or str(question).strip() == '':
                return "", ""
            answer, context = rag.answer_with_context(str(question))
            return answer, context
        except Exception as e:
            print(f"Error procesando pregunta: {e}")
            return f"ERROR: {str(e)}", ""
    
    # 5) Aplicamos con barras de progreso
    print("🔄 Generando respuestas para columna 'input'...")
    tqdm.pandas(desc="Procesando 'input'")
    
    # Aplicar la función que devuelve tupla (respuesta, contexto)
    results = df["input"].progress_apply(process_safely_with_context)
    
    # Separar respuestas y contextos
    df["actual_output"] = [result[0] for result in results]
    df["retrieval_context"] = [result[1] for result in results]
    
    if has_reformulated:
        print("🔄 Generando respuestas para columna 'input_reformulado_2'...")
        tqdm.pandas(desc="Procesando 'input_reformulado_2'")
        
        # Aplicar la función que devuelve tupla (respuesta, contexto)
        results_reformulated = df["input_reformulado_2"].progress_apply(process_safely_with_context)
        
        # Separar respuestas y contextos reformulados
        df["actual_output_reformulado"] = [result[0] for result in results_reformulated]
        df["retrieval_context_reformulado"] = [result[1] for result in results_reformulated]
    
    # 6) Guardamos el CSV con las nuevas columnas
    df.to_csv(output_path, index=False)
    print(f"✅ Respuestas generadas y guardadas en: {output_path}")
    
    # Mostrar estadísticas
    total_preguntas = len(df)
    respuestas_exitosas = len(df[df['actual_output'].str.contains('ERROR:', na=False) == False])
    print(f"📊 Estadísticas: {respuestas_exitosas}/{total_preguntas} respuestas generadas exitosamente")
    
    if has_reformulated:
        respuestas_reformuladas_exitosas = len(df[df['actual_output_reformulado'].str.contains('ERROR:', na=False) == False])
        print(f"📊 Reformuladas: {respuestas_reformuladas_exitosas}/{total_preguntas} respuestas generadas exitosamente")


class RAG:
    """
    Clase que encapsula el flujo RAG usando:
      1) Chunking de documentos.
      2) Indexación con FAISS + OllamaEmbeddings.
      3) LLM servido por vLLM a través de OpenAI API.
      4) Prompts en Español.
    """

    def __init__(
        self,
        docs: list[Document],
        embedding_model: str = "jina/jina-embeddings-v2-base-es:latest",    
        llm_model: str = "NousResearch/Meta-Llama-3-8B-Instruct",
        vllm_base_url: str = "http://localhost:8000/v1/",
        vllm_api_key: str = "not-needed",
        chunk_size: int = 512,
        chunk_overlap: int = 30,
        k: int = 5
    ):
        # 1) Split de documentos en chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunked_docs = splitter.split_documents(docs)

        # 2) Embeddings + FAISS index (usando Ollama para embeddings)
        embeddings = OllamaEmbeddings(
            model=embedding_model
        )
        self.db = FAISS.from_documents(chunked_docs, embeddings)
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        # 3) Cliente vLLM usando OpenAI API
        self.llm_client = OpenAI(
            base_url=vllm_base_url,
            api_key=vllm_api_key
        )
        self.llm_model = llm_model

        # 4) Prompt template
        self.prompt_template = """
        Contexto:
        {context}
        
        Pregunta:
        {question}
        
        Instrucciones:
        1. Básate UNICAMENTE en el contexto proporcionado.
        2. Responde UNICAMENTE a la pregunta planteada.
        3. La respuesta debe ser concisa, relevante y en español.
        4. Si la respuesta no puede deducirse del contexto, no escribas nada.
        5. No añadas explicaciones ni comentarios adicionales.
        
        Respuesta:
        """

    def _call_vllm(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """
        Llama al modelo vLLM usando la API de OpenAI.
        
        Args:
            prompt: El prompt completo para el modelo
            max_tokens: Número máximo de tokens a generar
            temperature: Temperatura para la generación
            
        Returns:
            str: Respuesta generada por el modelo
        """
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=None
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error llamando a vLLM: {e}")
            return f"ERROR: {str(e)}"

    def answer_with_context(self, question: str) -> tuple[str, str]:
        """
        1) Recupera los docs relevantes.
        2) Extrae y concatena su texto.
        3) Crea el prompt y llama a vLLM.
        4) Devuelve tanto la respuesta como el contexto usado.
        
        Returns:
            tuple: (respuesta, contexto_concatenado)
        """
        try:
            # Recuperar documentos relevantes
            relevant_docs = self.retriever.invoke(question)
            
            # Extraer y concatenar el contenido de los documentos
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Crear el prompt completo
            prompt = self.prompt_template.format(
                context=context_text,
                question=question
            )
            
            # Generar respuesta usando vLLM
            respuesta = self._call_vllm(prompt)
            
            return respuesta, context_text
            
        except Exception as e:
            print(f"Error en answer_with_context: {e}")
            return f"ERROR: {str(e)}", ""

    def answer(self, question: str) -> str:
        """
        Devuelve solo la respuesta.
        """
        respuesta, _ = self.answer_with_context(question)
        return respuesta


def main():
    """Función principal para pruebas independientes del módulo."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generar respuestas con RAG usando vLLM desde un CSV."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/jovyan/DEEPEVAL_AL/output/2_dataset_reformulado.csv",
        help="Ruta al archivo CSV que contiene las preguntas (y contexto)."
    )
    parser.add_argument(
        "--documents_dir",
        type=str,
        default="/home/jovyan/Documentos/Docs_pdf",
        help="Ruta a la carpeta con los PDFs a indexar."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/jovyan/DEEPEVAL_AL/output/3_dataset_reformulado_RAG_vllm.csv",
        help="Ruta donde se guardará el CSV resultante."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="jina/jina-embeddings-v2-base-es:latest",
        help="Modelo de embeddings para RAG (Ollama)."
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="NousResearch/Meta-Llama-3-8B-Instruct",
        help="Modelo LLM para RAG (vLLM)."
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://localhost:8000/v1/",
        help="URL base del servidor vLLM."
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default="not-needed",
        help="Clave API para vLLM."
    )
    
    args = parser.parse_args()
    
    # Llamar a la función principal
    generar_respuestas_rag(
        csv_path=args.csv_path,
        documents_dir=args.documents_dir,
        output_path=args.output_path,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        vllm_base_url=args.vllm_base_url,
        vllm_api_key=args.vllm_api_key
    )


if __name__ == "__main__":
    main()