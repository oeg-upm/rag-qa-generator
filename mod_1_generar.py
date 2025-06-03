"""
Clase para generar datasets sintéticos (pregunta-respuesta) a partir de documentos
usando DeepEval y un servidor vLLM local (OpenAI-compatible).
"""
import os
import glob
from pathlib import Path
import pandas as pd
from deepeval.synthesizer import Synthesizer
from deepeval.models import LocalModel
#from deepeval.models import OllamaModel
from deepeval.synthesizer.config import (
    StylingConfig, FiltrationConfig, EvolutionConfig, 
    Evolution, ContextConstructionConfig
)

class GeneradorPreguntas:
    """
    Clase para generar preguntas sintéticas a partir de documentos PDF
    usando DeepEval y un modelo vLLM local.
    """
    
    def __init__(self, 
                 model_name="NousResearch/Meta-Llama-3-8B-Instruct",
                 base_url="http://localhost:8000/v1/",
                 max_goldens_per_context=4,
                 num_evolutions=1,
                 max_contexts_per_document=3,
                 chunk_size=1024):
        """
        Inicializa el generador de preguntas.
        
        Args:
            model_name: Nombre del modelo vLLM
            base_url: URL base del servidor vLLM
            max_goldens_per_context: Máximo número de preguntas por contexto
            num_evolutions: Número de evoluciones para aumentar complejidad
            max_contexts_per_document: Máximo número de contextos por documento
            chunk_size: Tamaño de los chunks de texto
        """
        # Persistir base de vectores para acelerar embeddings
        os.environ["DEEPEVAL_PRESERVE_VECTOR_DB"] = "1"
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        
        self.model_name = model_name
        self.base_url = base_url
        self.max_goldens_per_context = max_goldens_per_context
        self.num_evolutions = num_evolutions
        self.max_contexts_per_document = max_contexts_per_document
        self.chunk_size = chunk_size
        
        # Configurar modelo vLLM
        self.model = LocalModel(
            model=model_name,
            #base_url=base_url,
            #openai_api_key=os.environ["OPENAI_API_KEY"]
        )
        '''
        # Initialize the custom model
        self.model = OllamaModel(
            model=model_name,
            base_url=base_url
        )
        '''
        print(f"🦙 Modelo configurado: {self.model.get_model_name()}")
        
        # Configurar synthesizer
        self._setup_synthesizer()
    
    def _setup_synthesizer(self):
        """Configura el synthesizer con las configuraciones necesarias."""
        
        # Configuración de Filtrado
        filtration_cfg = FiltrationConfig(
            synthetic_input_quality_threshold=0.5,
            max_quality_retries=3,
        )
        
        # Configuración de Evolución
        evolution_cfg = EvolutionConfig(
            num_evolutions=self.num_evolutions,
            evolutions={
                Evolution.REASONING: 1 / 7,
                Evolution.MULTICONTEXT: 1 / 7,
                Evolution.CONCRETIZING: 1 / 7,
                Evolution.CONSTRAINED: 1 / 7,
                Evolution.COMPARATIVE: 1 / 7,
                Evolution.HYPOTHETICAL: 1 / 7,
                Evolution.IN_BREADTH: 1 / 7
            }
        )
        
        # Configuración de Estilo en Español
        estilo_es = StylingConfig(
            # input_format: cómo deben formularse las entradas del usuario
            input_format=(
                "Genera preguntas claras, concisas y autónomas EN ESPAÑOL, "
                "que se puedan resolver únicamente con la información aportada "
                "en el contexto dado, sin requerir conocimientos externos."
            ),
            # expected_output_format: formato que debe tener la respuesta del sistema
            expected_output_format=(
                "Respuesta breve y precisa en ESPAÑOL. "
                "Devuelve UNICAMENTE la respuesta solicitada, "
                "sin explicaciones adicionales ni comentarios."
            ),
            # task: qué se espera que haga el sistema
            task=(
                "Atender y responder consultas específicas sobre el contenido "
                "de documentos proporcionados y en ESPAÑOL."
            ),
            # scenario: contexto y propósito de las entradas
            scenario=(
                "Evaluación de comprensión de textos en ESPAÑOL: "
                "un asistente que verifica la capacidad del usuario para "
                "extraer información concreta de documentos."
            )
        )
        
        # Crear synthesizer
        self.synthesizer = Synthesizer(
            model=self.model,
            async_mode=False,
            max_concurrent=5,
            filtration_config=filtration_cfg,
            evolution_config=evolution_cfg,
            styling_config=estilo_es,
            cost_tracking=True
        )
    
    def cargar_documentos(self, documents_dir):
        """
        Carga documentos desde un directorio.
        
        Args:
            documents_dir: Ruta al directorio con los documentos
            
        Returns:
            list: Lista de rutas a los documentos encontrados
        """
        documents_dir = Path(documents_dir)
        document_paths = []
        
        for ext in ("*.txt", "*.pdf", "*.docx"):
            document_paths += glob.glob(str(documents_dir / ext))
        
        print(f"📄 Documentos encontrados: {len(document_paths)}")
        return document_paths
    
    def generar_dataset(self, documents_dir, output_path):
        """
        Genera el dataset de preguntas y respuestas.
        
        Args:
            documents_dir: Directorio con los documentos fuente
            output_path: Ruta donde guardar el dataset generado
            
        Returns:
            str: Ruta al archivo generado
        """
        # Cargar documentos
        document_paths = self.cargar_documentos(documents_dir)
        
        if not document_paths:
            raise ValueError(f"No se encontraron documentos en {documents_dir}")
        
        # Configuración de construcción de contexto
        context_construction_cfg = ContextConstructionConfig(
            critic_model=self.model,
            max_contexts_per_document=self.max_contexts_per_document,
            min_contexts_per_document=1,
            max_context_length=3,
            min_context_length=1,
            chunk_size=self.chunk_size,
            chunk_overlap=0,
            context_quality_threshold=0.5,
            context_similarity_threshold=0.0,
            max_retries=3
        )
        
        # Generar goldens
        print("🤖 Generando preguntas y respuestas...")
        self.synthesizer.generate_goldens_from_docs(
            document_paths=document_paths,
            include_expected_output=True,
            max_goldens_per_context=self.max_goldens_per_context,
            context_construction_config=context_construction_cfg
        )
        
        print(f"✅ Goldens generados: {len(self.synthesizer.synthetic_goldens)}")
        
        # Convertir a DataFrame y guardar
        df = self.synthesizer.to_pandas()
        
        # Asegurar que el directorio de salida existe
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar dataset
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"💾 Dataset guardado en: {output_path}")
        
        return str(output_path)


def main():
    """Función principal para pruebas independientes del módulo."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar dataset de preguntas sintéticas')
    parser.add_argument('--documents_dir', type=str, 
                       default="/home/jovyan/Documentos/Docs_pdf",
                       help='Directorio con los documentos fuente')
    parser.add_argument('--output_path', type=str,
                       default="/home/jovyan/DEEPEVAL_AL/output/1_dataset.csv",
                       help='Ruta de salida del dataset')
    parser.add_argument('--model_name', type=str,
                       default="NousResearch/Meta-Llama-3-8B-Instruct",
                       help='Nombre del modelo vLLM')
    parser.add_argument('--base_url', type=str,
                       default="http://localhost:8000/v1/",
                       help='URL base del servidor vLLM')
    parser.add_argument('--max_goldens', type=int, default=4,
                       help='Máximo número de preguntas por contexto')
    
    args = parser.parse_args()
    
    # Crear generador y procesar
    generador = GeneradorPreguntas(
        model_name=args.model_name,
        base_url=args.base_url,
        max_goldens_per_context=args.max_goldens
    )
    
    generador.generar_dataset(args.documents_dir, args.output_path)


if __name__ == "__main__":
    main()