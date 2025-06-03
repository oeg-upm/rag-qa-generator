"""
Clase para reformular preguntas técnicas en versiones más accesibles
utilizando un modelo de lenguaje a través de VLLM y evaluando la calidad mediante
sentence embeddings y métricas ROUGE.
"""
import os
import pandas as pd
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer


class ReformuladorPreguntas:
    """
    Clase para reformular preguntas técnicas en lenguaje más accesible
    con evaluación de calidad semántica y léxica.
    """
    
    def __init__(self,
                 model_name="NousResearch/Meta-Llama-3-8B-Instruct",
                 base_url="http://localhost:8000/v1/",
                 api_key="not-needed",
                 umbral_similarity=0.75,
                 umbral_rouge_max=0.5,
                 max_intentos_reformulacion=3,
                 batch_size=5):
        """
        Inicializa el reformulador de preguntas.
        
        Args:
            model_name: Nombre del modelo vLLM
            base_url: URL base del servidor vLLM
            api_key: Clave API (ignorada por vLLM)
            umbral_similarity: Mínima similitud semántica requerida (0-1)
            umbral_rouge_max: Máxima similitud léxica permitida (0-1)
            max_intentos_reformulacion: Número máximo de intentos por pregunta
            batch_size: Tamaño del lote para procesamiento
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.umbral_similarity = umbral_similarity
        self.umbral_rouge_max = umbral_rouge_max
        self.max_intentos_reformulacion = max_intentos_reformulacion
        self.batch_size = batch_size
        
        # Configurar cliente OpenAI para vLLM
        self.cliente = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        # Inicializar modelo de embeddings y evaluador ROUGE
        self.modelo_embeddings = None
        self.rouge_evaluador = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        print(f"🦙 Reformulador configurado con modelo: {self.model_name}")
    
    def _cargar_modelo_embeddings(self):
        """
        Carga el modelo de sentence embeddings para español.
        
        Returns:
            SentenceTransformer: Modelo cargado
        """
        try:
            # Intenta cargar primero un modelo multilingüe optimizado para español
            return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        except Exception as e:
            print(f"Error al cargar el modelo principal, intentando alternativa: {e}")
            # Alternativa si el primer modelo falla
            return SentenceTransformer('distiluse-base-multilingual-cased-v1')
    
    def detectar_campo(self, pregunta):
        """
        Detecta el campo o tema al que pertenece una pregunta.
        
        Args:
            pregunta: Texto de la pregunta
            
        Returns:
            str: Campo o área temática detectada
        """
        prompt = f"""
        En Español. Analiza la siguiente pregunta y determina a qué campo o área temática pertenece.
        Responde ÚNICAMENTE con el nombre del campo (por ejemplo: "Medicina", "Derecho", "Tecnología", etc.). 
        No incluyas explicaciones adicionales.
        
        Pregunta: {pregunta} 
        
        Campo: 
        """
        
        try:
            respuesta = self.cliente.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0
            )
            return respuesta.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error al detectar campo: {e}")
            return "General"
    
    def reformular(self, pregunta, campo):
        """
        Reformula una pregunta técnica en un lenguaje más accesible.
        
        Args:
            pregunta: Texto de la pregunta original
            campo: Campo o área temática de la pregunta
            
        Returns:
            str: Pregunta reformulada
        """
        prompt = f"""
        En Español. Eres un modelo que escribe y habla en español. 
        Necesito que reformules una pregunta técnica de {campo} para hacerla más comprensible 
        para una persona sin conocimientos especializados en ese campo.
        
        Reglas importantes:
        1. La reformulación debe mantener EXACTAMENTE el mismo significado e intención que la original (alta similitud semántica o "sentence similarity")
        2. Debes usar palabras diferentes y estructura de frase distinta (bajo solapamiento léxico o bajo valor de "ROUGE")
        3. Cambia los términos técnicos por explicaciones simples o analogías
        4. La persona no tiene conocimientos sobre {campo}
        5. Acorta la longitud de la pregunta guardando el mismo significado
        6. Si es necesario divide la pregunta en dos preguntas más simples
        7. Sé lo menos técnico posible
        8. Responde ÚNICAMENTE con la pregunta reformulada, sin añadir comentarios
        9. Es IMPORTANTE que no incluyas explicaciones adicionales.

        Esta es la pregunta original: {pregunta}
        
        Pregunta reformulada:
        """
        
        try:
            respuesta = self.cliente.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.7
            )
            return respuesta.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error al reformular pregunta: {e}")
            return pregunta
    
    def simplificar(self, pregunta, campo):
        """
        Reformula una pregunta ya reformulada para hacerla aún más simple y corta.
        
        Args:
            pregunta: Texto de la pregunta reformulada
            campo: Campo o área temática de la pregunta
            
        Returns:
            str: Pregunta simplificada
        """
        prompt = f"""
        En Español. Eres un experto en simplificar y acortar preguntas.
        
        Tengo una pregunta ya reformulada sobre {campo}, pero necesito que sea aún más corta y simple.
        
        Reglas importantes:
        1. La versión simplificada DEBE mantener el mismo significado que la pregunta original (alta similitud semántica o "sentence similarity")
        2. Usa vocabulario y estructura TOTALMENTE DIFERENTES (bajo solapamiento léxico o bajo valor de "ROUGE")
        3. Reduce la longitud a menos de la mitad sin perder el significado esencial
        4. Usa palabras más sencillas y frases más directas
        5. Elimina cualquier explicación o contexto innecesario
        6. Mantén la pregunta clara y comprensible
        7. Responde ÚNICAMENTE con la pregunta simplificada, sin añadir comentarios
        8. Es IMPORTANTE que no incluyas explicaciones adicionales.
        
        Pregunta reformulada: {pregunta}
        
        Pregunta simplificada:
        """
        
        try:
            respuesta = self.cliente.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.8
            )
            return respuesta.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error al simplificar pregunta: {e}")
            return pregunta
    
    def calcular_similitud_semantica(self, texto1, texto2):
        """
        Calcula la similitud semántica entre dos textos utilizando sentence embeddings.
        
        Args:
            texto1: Primer texto
            texto2: Segundo texto
            
        Returns:
            float: Valor de similitud (0-1)
        """
        # Cargar el modelo si aún no está inicializado
        if self.modelo_embeddings is None:
            self.modelo_embeddings = self._cargar_modelo_embeddings()
        
        try:
            # Codificar los textos
            embedding1 = self.modelo_embeddings.encode(texto1, convert_to_tensor=True)
            embedding2 = self.modelo_embeddings.encode(texto2, convert_to_tensor=True)
            
            # Calcular similitud de coseno
            similitud = util.pytorch_cos_sim(embedding1, embedding2).item()
            
            return similitud
        except Exception as e:
            print(f"Error al calcular similitud semántica: {e}")
            return 0.0
    
    def calcular_rouge(self, texto1, texto2):
        """
        Calcula el valor ROUGE entre dos textos para medir la similitud léxica.
        
        Args:
            texto1: Texto original
            texto2: Texto reformulado
            
        Returns:
            float: Promedio de puntuaciones ROUGE (0-1)
        """
        try:
            scores = self.rouge_evaluador.score(texto1, texto2)
            
            # Calcular el promedio de varios tipos de ROUGE
            rouge_promedio = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
            
            return rouge_promedio
        except Exception as e:
            print(f"Error al calcular ROUGE: {e}")
            return 1.0  # En caso de error, asumimos máxima similitud (peor caso)
    
    def evaluar_calidad_reformulacion(self, pregunta_original, pregunta_reformulada):
        """
        Evalúa si la pregunta reformulada mantiene la semántica pero cambia el léxico.
        
        Args:
            pregunta_original: Texto de la pregunta original
            pregunta_reformulada: Texto de la pregunta reformulada
            
        Returns:
            bool: True si la reformulación es válida, False en caso contrario
            dict: Diccionario con métricas detalladas
        """
        # Calcular similitud semántica
        similitud = self.calcular_similitud_semantica(pregunta_original, pregunta_reformulada)
        
        # Calcular similitud léxica (ROUGE)
        rouge = self.calcular_rouge(pregunta_original, pregunta_reformulada)
        
        # Una buena reformulación tiene alta similitud semántica y baja similitud léxica
        es_valida = (similitud >= self.umbral_similarity) and (rouge <= self.umbral_rouge_max)
        
        metricas = {
            'similitud_semantica': similitud,
            'similitud_lexica': rouge,
            'es_valida': es_valida
        }
        
        return es_valida, metricas
    
    def procesar_dataset(self, input_path, output_path):
        """
        Procesa un archivo CSV con preguntas y añade versiones reformuladas.
        
        Args:
            input_path: Ruta al archivo CSV de entrada
            output_path: Ruta donde guardar el archivo CSV de salida
        """
        print(f"📂 Cargando dataset desde: {input_path}")
        
        # Leer el CSV de entrada
        try:
            df = pd.read_csv(input_path)
            print(f"📂 CSV cargado correctamente. Columnas: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error al leer el archivo CSV: {e}")
            return
        
        # Verificar que existe la columna 'input'
        if 'input' not in df.columns:
            print("Error: El archivo CSV no contiene una columna 'input'")
            return
        
        # Crear columnas para preguntas reformuladas y métricas si no existen
        # Definir todas las columnas necesarias y sus valores por defecto
        columnas_config = {
            'campo_tematico': '',
            'input_reformulado_1': '',
            'input_reformulado_2': '',
            'similitud_semantica_1': 0.0,
            'similitud_lexica_1': 0.0,
            'intentos_reformulacion_1': 0,
            'similitud_semantica_2': 0.0,
            'similitud_lexica_2': 0.0,
            'intentos_reformulacion_2': 0,
            'actual_output': '',
            'actual_output_reformulado': '',
            'expected_output': '',
            'context': '',
            'retrieval_context': '',
            'retrieval_context_reformulado': '',
            'n_chunks_per_context': 0,
            'context_length': 0,
            'evolutions': '',
            'context_quality': 0.0,
            'synthetic_input_quality': 0.0,
            'source_file': ''
        }
        
        # Crear columnas que no existen
        for columna, valor_defecto in columnas_config.items():
            if columna not in df.columns:
                df[columna] = valor_defecto
    
        
        total_preguntas = len(df)
        
        # PRIMERA REFORMULACIÓN
        print(f"🌟 Procesando {total_preguntas} preguntas reformulación...")
        
        for i in tqdm(range(0, total_preguntas, self.batch_size), desc="Procesando lotes (reformulación)"):
            lote = df.iloc[i:min(i+self.batch_size, total_preguntas)]
            
            for idx, fila in lote.iterrows():
                pregunta = fila['input']
                
                
                if pd.notna(df.at[idx, 'input_reformulado_1']) and df.at[idx, 'input_reformulado_1'] != '':
                    continue
                    
                # Detectar campo de la pregunta
                campo = self.detectar_campo(pregunta)
                df.at[idx, 'campo_tematico'] = campo
                
                # Proceso de reformulación con validación de calidad
                reformulada = ""
                es_valida = False
                metricas = {}
                intentos = 0
                
                while not es_valida and intentos < self.max_intentos_reformulacion:
                    intentos += 1
                    
                    # Reformular la pregunta
                    reformulada = self.reformular(pregunta, campo)
                    
                    # Evaluar calidad de la reformulación
                    es_valida, metricas = self.evaluar_calidad_reformulacion(pregunta, reformulada)
                    '''
                    if es_valida:
                        print(f"✨ Pregunta {idx} reformulada válidamente en intento {intentos} – Similitud semántica: {metricas['similitud_semantica']:.2f}, Similitud léxica: {metricas['similitud_lexica']:.2f}")
                        break
                    else:
                        print(f"❌ Intento {intentos}: Reformulación inválida – Similitud semántica: {metricas['similitud_semantica']:.2f} (mín: {self.umbral_similarity}), Similitud léxica: {metricas['similitud_lexica']:.2f} (máx: {self.umbral_rouge_max})")
                    '''
                # Guardar resultados
                if not es_valida:
                    #print(f"⚠️ ADVERTENCIA: No se logró una reformulación válida para la pregunta {idx} después de {self.max_intentos_reformulacion} intentos")
                    df.at[idx, 'intentos_reformulacion_1'] = 0
                else:
                    df.at[idx, 'intentos_reformulacion_1'] = intentos
                
                df.at[idx, 'input_reformulado_1'] = reformulada
                df.at[idx, 'similitud_semantica_1'] = round(metricas.get('similitud_semantica', 0.0), 2)
                df.at[idx, 'similitud_lexica_1'] = round(metricas.get('similitud_lexica', 1.0), 2)
                
                # Guardar progreso incremental
                if (idx % self.batch_size == 0) or (idx == total_preguntas - 1):
                    df.to_csv(output_path, index=False)
        
        # SEGUNDA REFORMULACIÓN
        print(f"🌟 Procesando {total_preguntas} preguntas simplificación...")
        
        for i in tqdm(range(0, total_preguntas, self.batch_size), desc="Procesando lotes (simlificación)"):
            lote = df.iloc[i:min(i+self.batch_size, total_preguntas)]
            
            for idx, fila in lote.iterrows():
                # Solo procesar si ya existe una reformulación previa y falta la segunda
                if not pd.notna(fila['input_reformulado_1']) or fila['input_reformulado_1'] == '':
                    continue
                    
                
                if pd.notna(df.at[idx, 'input_reformulado_2']) and df.at[idx, 'input_reformulado_2'] != '':
                    continue
                    
                pregunta_original = fila['input']
                pregunta_reformulada = fila['input_reformulado_1']
                campo = fila['campo_tematico']
                
                # Proceso de segunda reformulación
                simplificada = ""
                es_valida = False
                metricas = {}
                intentos = 0
                
                while not es_valida and intentos < self.max_intentos_reformulacion:
                    intentos += 1
                    
                    # Simplificar la pregunta
                    simplificada = self.simplificar(pregunta_reformulada, campo)
                    
                    # Evaluar calidad (comparando con la pregunta original)
                    es_valida, metricas = self.evaluar_calidad_reformulacion(pregunta_original, simplificada)
                    '''
                    if es_valida:
                        print(f"✨ Pregunta {idx} simplificada válidamente en intento {intentos} – Similitud semántica: {metricas['similitud_semantica']:.2f}, Similitud léxica: {metricas['similitud_lexica']:.2f}")
                        break
                    else:
                        print(f"❌ Intento {intentos}: Simplificación inválida – Similitud semántica: {metricas['similitud_semantica']:.2f} (mín: {self.umbral_similarity}), Similitud léxica: {metricas['similitud_lexica']:.2f} (máx: {self.umbral_rouge_max})")
                    '''
                
                # Guardar resultados
                if not es_valida:
                    #print(f"⚠️ ADVERTENCIA: No se logró una simplificación válida para la pregunta {idx} después de {self.max_intentos_reformulacion} intentos")
                    df.at[idx, 'intentos_reformulacion_2'] = 0
                else:
                    df.at[idx, 'intentos_reformulacion_2'] = intentos
                
                df.at[idx, 'input_reformulado_2'] = simplificada
                df.at[idx, 'similitud_semantica_2'] = round(metricas.get('similitud_semantica', 0.0), 2)
                df.at[idx, 'similitud_lexica_2'] = round(metricas.get('similitud_lexica', 1.0), 2)
                
                # Guardar progreso incremental
                if (idx % self.batch_size == 0) or (idx == total_preguntas - 1):
                    df.to_csv(output_path, index=False)
    
    
        # Reordenar columnas para un orden lógico
        columnas_deseadas = [
            'input', 
            'campo_tematico', 
            'similitud_semantica_1', 
            'similitud_lexica_1', 
            'intentos_reformulacion_1', 
            'input_reformulado_1', 
            'similitud_semantica_2', 
            'similitud_lexica_2', 
            'intentos_reformulacion_2', 
            'input_reformulado_2', 
            'actual_output', 
            'actual_output_reformulado', 
            'expected_output', 
            'context', 
            'retrieval_context',
            'retrieval_context_reformulado',
            'n_chunks_per_context', 
            'context_length', 
            'evolutions', 
            'context_quality', 
            'synthetic_input_quality', 
            'source_file'
        ]
        
        # Añadir columnas adicionales que puedan existir
        otras_columnas = [col for col in df.columns if col not in columnas_deseadas]
        orden_final = columnas_deseadas + otras_columnas
        
        # Reordenar el dataframe
        df = df[orden_final]
        df.to_csv(output_path, index=False)
        print(f"✅ Proceso completado. Dataset reformulado guardado en: {output_path}")


def main():
    """Función principal para pruebas independientes del módulo."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reformular preguntas técnicas en versiones más comprensibles')
    parser.add_argument('--input', type=str, 
                       default="/home/jovyan/DEEPEVAL_AL/output/1_dataset.csv", 
                       help='Ruta al archivo CSV de entrada')
    parser.add_argument('--output', type=str, 
                       default="/home/jovyan/DEEPEVAL_AL/output/2_dataset_reformulado.csv", 
                       help='Ruta donde guardar el archivo CSV de salida')
    parser.add_argument('--model_name', type=str,
                       default="NousResearch/Meta-Llama-3-8B-Instruct",
                       help='Nombre del modelo vLLM')
    parser.add_argument('--base_url', type=str,
                       default="http://localhost:8000/v1/",
                       help='URL base del servidor vLLM')
    parser.add_argument('--batch', type=int, default=5, 
                       help='Tamaño del lote para procesamiento')
    parser.add_argument('--max_intentos', type=int, default=3, 
                       help='Número máximo de intentos para reformular una pregunta')
    parser.add_argument('--sim_umbral', type=float, default=0.75, 
                       help='Umbral mínimo de similitud semántica (0-1)')
    parser.add_argument('--rouge_umbral', type=float, default=0.5, 
                       help='Umbral máximo de similitud léxica (0-1)')
    
    args = parser.parse_args()
    
    # Crear reformulador y procesar
    reformulador = ReformuladorPreguntas(
        model_name=args.model_name,
        base_url=args.base_url,
        umbral_similarity=args.sim_umbral,
        umbral_rouge_max=args.rouge_umbral,
        max_intentos_reformulacion=args.max_intentos,
        batch_size=args.batch
    )
    
    reformulador.procesar_dataset(args.input, args.output)


if __name__ == "__main__":
    main()