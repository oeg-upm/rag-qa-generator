import pandas as pd
from pathlib import Path
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models import LocalModel
from deepeval.metrics import (
    AnswerRelevancyMetric, 
    FaithfulnessMetric, 
    ContextualPrecisionMetric, 
    ContextualRecallMetric, 
    ContextualRelevancyMetric
)
import ast
import logging
import time
from typing import Dict, List, Optional, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimultaneousRowEvaluator:
    """
    Clase para evaluar filas simultáneamente en ambos datasets (original y reformulado).
    Solo avanza a la siguiente fila cuando ambas evaluaciones son exitosas.
    """
    
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", threshold=0.5, max_retries=10):
        """
        Inicializa el evaluador simultáneo.
        
        Args:
            model_name (str): Nombre del modelo a usar
            threshold (float): Umbral para las métricas
            max_retries (int): Número máximo de reintentos por métrica
        """
        self.model_name = model_name
        self.threshold = threshold
        self.max_retries = max_retries
        self.metric_names = ['contextual_precision', 'contextual_recall', 'contextual_relevancy', 
                           'answer_relevancy', 'faithfulness']
        
    def setup_model(self):
        """Configura el modelo vLLM para las métricas."""
        logger.info(f"Configurando modelo: {self.model_name}")
        return LocalModel(model=self.model_name)
    
    def setup_metrics(self, model):
        """
        Configura todas las métricas de evaluación.
        CREA NUEVAS INSTANCIAS PARA CADA EVALUACIÓN
        
        Args:
            model: Modelo a usar para las métricas
            
        Returns:
            dict: Diccionario con las métricas configuradas
        """
        logger.debug("Configurando métricas de evaluación")
        
        metrics = {
            'contextual_precision': ContextualPrecisionMetric(
                threshold=self.threshold,
                model=model,
                include_reason=True,
                async_mode=False
            ),
            'contextual_recall': ContextualRecallMetric(
                threshold=self.threshold,
                model=model,
                include_reason=True,
                async_mode=False
            ),
            'contextual_relevancy': ContextualRelevancyMetric(
                threshold=self.threshold,
                model=model,
                include_reason=True,
                async_mode=False
            ),
            'answer_relevancy': AnswerRelevancyMetric(
                threshold=self.threshold,
                model=model,
                include_reason=True,
                async_mode=False
            ),
            'faithfulness': FaithfulnessMetric(
                threshold=self.threshold,
                model=model,
                include_reason=True,
                async_mode=False
            )
        }
        
        return metrics
    
    def parse_context(self, context_str):
        """
        Parsea el contexto desde string a lista.
        
        Args:
            context_str: String que representa el contexto
            
        Returns:
            list: Lista con el contexto parseado
        """
        if pd.isna(context_str) or context_str == '':
            return []
        
        try:
            # Intentar parsear como lista de Python
            if context_str.startswith('[') and context_str.endswith(']'):
                return ast.literal_eval(context_str)
            else:
                # Si no es una lista, tratarlo como string simple
                return [str(context_str)]
        except:
            # Si falla el parsing, tratarlo como string
            return [str(context_str)]
    
    def create_test_case(self, row_data):
        """
        Crea un caso de prueba desde una fila de datos.
        
        Args:
            row_data (dict): Datos de la fila
            
        Returns:
            LLMTestCase: Caso de prueba creado
        """
        return LLMTestCase(
            input=str(row_data['input']),
            actual_output=str(row_data['actual_output']),
            expected_output=str(row_data['expected_output']) if row_data['expected_output'] else "",
            context=self.parse_context(row_data['context']),
            retrieval_context=self.parse_context(row_data['retrieval_context'])
        )
    
    def evaluate_single_metric(self, test_case, metric_name, metric_instance):
        """
        Evalúa una sola métrica para un caso de prueba.
        
        Args:
            test_case: Caso de prueba
            metric_name (str): Nombre de la métrica
            metric_instance: Instancia de la métrica
            
        Returns:
            tuple: (score, reason) o (None, None) si hay error
        """
        try:
            # Evaluar solo este test_case con esta métrica
            evaluate(
                [test_case], 
                [metric_instance], 
                ignore_errors=False,
                run_async=False,
                print_results=False,
                write_cache=False,
                use_cache=False,
                show_indicator=False,
                verbose_mode=False
            )
            
            # Extraer resultados
            score = getattr(metric_instance, 'score', None)
            reason = getattr(metric_instance, 'reason', None)
            
            # Manejar listas
            if isinstance(score, list) and len(score) > 0:
                score = score[0]
            if isinstance(reason, list) and len(reason) > 0:
                reason = reason[0]
            
            return score, reason
            
        except Exception as e:
            logger.warning(f"Error evaluando métrica {metric_name}: {e}")
            return None, None
    
    def evaluate_dataset_version(self, row_data, version_name):
        """
        Evalúa una versión del dataset (original o reformulado) para una fila.
        FALLA RÁPIDO: En cuanto una métrica falla, abandona la evaluación de esta versión.
        
        Args:
            row_data (dict): Datos de la fila
            version_name (str): Nombre de la versión ("original" o "reformulado")
            
        Returns:
            tuple: (success, results) donde success indica si todas las métricas fueron exitosas
        """
        logger.info(f"    Evaluando versión {version_name}")
        
        # Crear caso de prueba
        test_case = self.create_test_case(row_data)
        
        # Inicializar resultados
        results = {
            'input': row_data['input'],
            'actual_output': row_data['actual_output'],
            'expected_output': row_data['expected_output'],
            'context': row_data['context'],
            'retrieval_context': row_data['retrieval_context']
        }
        
        # Evaluar cada métrica con reintentos - FALLA RÁPIDO
        for metric_name in self.metric_names:
            success = False
            attempt = 0
            
            while not success and attempt < self.max_retries:
                attempt += 1
                logger.info(f"      {metric_name} - Intento {attempt}/{self.max_retries}")
                
                try:
                    # Crear nueva instancia del modelo y métrica para este intento
                    model = self.setup_model()
                    metrics = self.setup_metrics(model)
                    metric_instance = metrics[metric_name]
                    
                    # Evaluar métrica
                    score, reason = self.evaluate_single_metric(test_case, metric_name, metric_instance)
                    
                    if score is not None:
                        results[f'{metric_name}_score'] = score
                        results[f'{metric_name}_reason'] = reason
                        success = True
                        logger.info(f"        ✅ {metric_name} exitoso (score: {score})")
                    else:
                        logger.warning(f"        🔄 {metric_name} falló en intento {attempt}")
                        if attempt < self.max_retries:
                            time.sleep(2)
                        
                except Exception as e:
                    logger.error(f"        ❌ Error en {metric_name} intento {attempt}: {e}")
                    if attempt < self.max_retries:
                        time.sleep(2)
            
            # Si no tuvo éxito después de todos los intentos - FALLA RÁPIDO
            if not success:
                logger.error(f"        ⚠️ {metric_name} falló - ABANDONANDO evaluación de versión {version_name}")
                return False, None  # Falla rápido - no continuar con más métricas
        
        # Si llegamos aquí, todas las métricas fueron exitosas
        return True, results
    
    def evaluate_row_simultaneously(self, row_idx, original_data, reformulated_data):
        """
        Evalúa una fila simultáneamente para ambos datasets.
        FALLA RÁPIDO: Si cualquier métrica de cualquier versión falla, abandona toda la fila.
        
        Args:
            row_idx (int): Índice de la fila
            original_data (dict): Datos de la versión original
            reformulated_data (dict): Datos de la versión reformulada
            
        Returns:
            tuple: (success, original_results, reformulated_results)
        """
        logger.info(f"  Evaluando fila {row_idx + 1} - AMBAS VERSIONES")
        
        # Evaluar versión original - FALLA RÁPIDO
        logger.info(f"    🔄 Comenzando evaluación ORIGINAL...")
        original_success, original_results = self.evaluate_dataset_version(original_data, "original")
        
        if not original_success:
            logger.warning(f"  ❌ Fila {row_idx + 1} FALLÓ en versión ORIGINAL - ABANDONANDO fila completa")
            return False, None, None
        
        logger.info(f"    ✅ Versión ORIGINAL exitosa - Continuando con REFORMULADA...")
        
        # Solo evaluar reformulada si original fue exitosa
        reformulated_success, reformulated_results = self.evaluate_dataset_version(reformulated_data, "reformulado")
        
        if not reformulated_success:
            logger.warning(f"  ❌ Fila {row_idx + 1} FALLÓ en versión REFORMULADA - ABANDONANDO fila completa")
            return False, None, None
        
        # Si llegamos aquí, ambas versiones fueron exitosas
        logger.info(f"  🎉 Fila {row_idx + 1} COMPLETAMENTE EXITOSA - Ambas versiones OK")
        return True, original_results, reformulated_results

def load_dataset(file_path):
    """
    Carga el dataset desde un archivo CSV.
    
    Args:
        file_path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    logger.info(f"Cargando dataset desde: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset cargado exitosamente. Filas disponibles: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise

def initialize_csv_with_headers(output_filename, metric_names):
    """
    Inicializa el archivo CSV con los headers si no existe.
    
    Args:
        output_filename (str): Nombre del archivo de salida
        metric_names (list): Lista de nombres de métricas
    """
    if not Path(output_filename).exists():
        headers = ['input', 'actual_output', 'expected_output', 'context', 'retrieval_context']
        for metric_name in metric_names:
            headers.extend([f'{metric_name}_score', f'{metric_name}_reason'])
        
        # Crear DataFrame vacío con headers y guardarlo
        empty_df = pd.DataFrame(columns=headers)
        empty_df.to_csv(output_filename, index=False)
        logger.info(f"Archivo CSV inicializado con headers: {output_filename}")

def append_row_to_csv(output_filename, row_result):
    """
    Añade una fila de resultados al archivo CSV de forma incremental.
    
    Args:
        output_filename (str): Nombre del archivo de salida
        row_result (dict): Diccionario con los resultados de la fila
    """
    try:
        # Convertir el resultado a DataFrame de una fila
        row_df = pd.DataFrame([row_result])
        
        # Añadir al archivo CSV existente
        row_df.to_csv(output_filename, mode='a', header=False, index=False)
        
    except Exception as e:
        logger.error(f"Error guardando fila incremental: {e}")
        raise

def get_processed_rows_count(output_filename):
    """
    Obtiene el número de filas ya procesadas en el archivo CSV.
    
    Args:
        output_filename (str): Nombre del archivo de salida
        
    Returns:
        int: Número de filas procesadas (excluyendo header)
    """
    try:
        if Path(output_filename).exists():
            df = pd.read_csv(output_filename)
            return len(df)
        else:
            return 0
    except Exception as e:
        logger.warning(f"Error contando filas procesadas: {e}")
        return 0

def process_simultaneous_evaluation(df, target_successful_rows, output_original, output_reformulado, 
                                  model_name, threshold, max_retries=10):
    """
    Procesa evaluación simultánea hasta obtener el número objetivo de filas exitosas.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        target_successful_rows (int): Número objetivo de filas completamente exitosas
        output_original (str): Archivo de salida para datos originales
        output_reformulado (str): Archivo de salida para datos reformulados
        model_name (str): Nombre del modelo
        threshold (float): Umbral para las métricas
        max_retries (int): Número máximo de reintentos por métrica
    """
    logger.info(f"Procesando evaluación simultánea")
    logger.info(f"Objetivo: {target_successful_rows} filas completamente exitosas")
    logger.info(f"Configuración: max_retries={max_retries}, threshold={threshold}")
    
    # Crear evaluador
    evaluator = SimultaneousRowEvaluator(model_name=model_name, threshold=threshold, max_retries=max_retries)
    
    # Inicializar archivos CSV con headers
    initialize_csv_with_headers(output_original, evaluator.metric_names)
    initialize_csv_with_headers(output_reformulado, evaluator.metric_names)
    
    # Verificar cuántas filas exitosas ya tenemos
    current_successful_original = get_processed_rows_count(output_original)
    current_successful_reformulated = get_processed_rows_count(output_reformulado)
    current_successful = min(current_successful_original, current_successful_reformulated)
    
    logger.info(f"Filas exitosas ya procesadas: {current_successful}")
    
    if current_successful >= target_successful_rows:
        logger.info("Ya se alcanzó el objetivo de filas exitosas")
        return
    
    # Obtener filas válidas para evaluar
    valid_rows = []
    for idx, row in df.iterrows():
        # Verificar que ambas versiones tengan datos válidos
        original_valid = not (pd.isna(row['input']) or pd.isna(row['actual_output']))
        reformulated_valid = not (pd.isna(row['input_reformulado_2']) or pd.isna(row['actual_output_reformulado']))
        
        if original_valid and reformulated_valid:
            original_data = {
                'input': row['input'],
                'actual_output': row['actual_output'],
                'expected_output': row['expected_output'] if pd.notna(row['expected_output']) else "",
                'context': row['context'],
                'retrieval_context': row['retrieval_context']
            }
            
            reformulated_data = {
                'input': row['input_reformulado_2'],
                'actual_output': row['actual_output_reformulado'],
                'expected_output': row['expected_output'] if pd.notna(row['expected_output']) else "",
                'context': row['context'],
                'retrieval_context': row['retrieval_context_reformulado']
            }
            
            valid_rows.append((idx, original_data, reformulated_data))
        else:
            logger.warning(f"Saltando fila {idx}: datos inválidos en alguna versión")
    
    if not valid_rows:
        logger.warning("No se encontraron filas válidas para evaluar")
        return
    
    logger.info(f"Filas válidas encontradas: {len(valid_rows)}")
    
    # Comenzar evaluación desde donde se quedó
    successful_rows = current_successful
    current_row_index = 0
    total_attempts = 0
    
    while successful_rows < target_successful_rows and current_row_index < len(valid_rows):
        original_idx, original_data, reformulated_data = valid_rows[current_row_index]
        total_attempts += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"INTENTO {total_attempts} - FILA DATASET {original_idx} - OBJETIVO: {successful_rows + 1}/{target_successful_rows}")
        logger.info(f"{'='*80}")
        
        try:
            # Evaluar ambas versiones simultáneamente
            row_success, original_results, reformulated_results = evaluator.evaluate_row_simultaneously(
                current_row_index, original_data, reformulated_data
            )
            
            if row_success:
                # Ambas versiones fueron exitosas - guardar ambos resultados
                append_row_to_csv(output_original, original_results)
                append_row_to_csv(output_reformulado, reformulated_results)
                
                successful_rows += 1
                logger.info(f"🎉 FILA EXITOSA GUARDADA - Progreso: {successful_rows}/{target_successful_rows}")
                
                # Avanzar a la siguiente fila del dataset
                current_row_index += 1
            else:
                # Al menos una versión falló - intentar con la siguiente fila
                current_row_index += 1
                logger.warning(f"🔄 Fila falló, probando con la siguiente fila del dataset")
                
        except Exception as e:
            logger.error(f"Error crítico procesando fila: {e}")
            # Avanzar a la siguiente fila en caso de error crítico
            current_row_index += 1
            
        # Verificar si se agotaron las filas del dataset
        if current_row_index >= len(valid_rows):
            logger.warning(f"Se agotaron las filas del dataset. Solo se lograron {successful_rows} filas exitosas de {target_successful_rows}")
            break
    
    # Estadísticas finales
    logger.info(f"\n{'='*80}")
    logger.info(f"ESTADÍSTICAS FINALES")
    logger.info(f"{'='*80}")
    logger.info(f"Filas exitosas obtenidas: {successful_rows}/{target_successful_rows}")
    logger.info(f"Total de intentos realizados: {total_attempts}")
    logger.info(f"Filas del dataset evaluadas: {current_row_index}/{len(valid_rows)}")
    logger.info(f"Archivos generados:")
    logger.info(f"  - Original: {output_original}")
    logger.info(f"  - Reformulado: {output_reformulado}")
    
    if successful_rows >= target_successful_rows:
        logger.info("🎉 ¡OBJETIVO ALCANZADO!")
    else:
        logger.warning("⚠️ No se pudo alcanzar el objetivo completo")

def main():
    """
    Función principal que ejecuta todo el proceso de evaluación simultánea.
    """
    # Configuración
    output_dir = Path("/home/jovyan/DEEPEVAL_AL/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    CSV_PATH = output_dir / "a3_dataset_reformulado_RAG.csv"
    OUTPUT_ORIGINAL = output_dir / "a4_evaluacion_original_simultanea.csv"
    OUTPUT_REFORMULADO = output_dir / "a4_evaluacion_reformulado_simultanea.csv"
    
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    THRESHOLD = 0.0
    TARGET_SUCCESSFUL_ROWS = 40  # Objetivo: 20 filas completamente exitosas
    MAX_RETRIES = 1  # Número máximo de reintentos por métrica
    
    try:
        # Cargar dataset completo
        df = load_dataset(CSV_PATH)
        
        # Procesar evaluación simultánea
        logger.info("\n" + "="*80)
        logger.info("INICIANDO EVALUACIÓN SIMULTÁNEA")
        logger.info("="*80)
        
        process_simultaneous_evaluation(
            df=df,
            target_successful_rows=TARGET_SUCCESSFUL_ROWS,
            output_original=OUTPUT_ORIGINAL,
            output_reformulado=OUTPUT_REFORMULADO,
            model_name=MODEL_NAME,
            threshold=THRESHOLD,
            max_retries=MAX_RETRIES
        )
        
        logger.info("\n" + "="*80)
        logger.info("PROCESO COMPLETADO")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error en el proceso principal: {e}")
        raise

if __name__ == "__main__":
    main()