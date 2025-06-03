"""
Script principal para ejecutar el pipeline completo de RAG:
  1) Generar preguntas sintéticas desde documentos PDF (Módulo 1)
  2) Reformular preguntas para hacerlas más accesibles (Módulo 2)  
  3) Construir el índice RAG y generar respuestas (Módulo 3)
  4) Guardar el dataset final con preguntas y respuestas
"""
import argparse
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# Importar las clases de los módulos
from mod_1_generar import GeneradorPreguntas
from mod_2_reformular import ReformuladorPreguntas
from mod_3_RAG_vllm import generar_respuestas_rag


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo de generación de dataset RAG."
    )
    
    # Argumentos de entrada y salida
    parser.add_argument(
        "--documents_dir",
        type=str,
        default="/home/jovyan/Documentos/Docs_txt",
        help="Ruta a la carpeta con los PDFs a procesar."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jovyan/DEEPEVAL_AL/output",
        help="Directorio donde se guardarán todos los archivos de salida."
    )
    
    # Argumentos del Módulo 1 (Generación de preguntas)
    parser.add_argument(
        "--model_name_gen",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        #default="NousResearch/Meta-Llama-3-8B-Instruct",
        #default = "deepseek-r1:1.5b",
        #default="llama3.1:8b-instruct-fp16",
        help="Nombre del modelo vLLM para generación de preguntas."
    )
    parser.add_argument(
        "--base_url_gen",
        type=str,
        default="http://localhost:8000/v1/",
        #default="http://localhost:11434",
        help="URL base del servidor vLLM para generación."
    )
    parser.add_argument(
        "--max_goldens_per_context",
        type=int,
        default=2,
        help="Máximo número de preguntas por contexto."
    )
    parser.add_argument(
        "--num_evolutions",
        type=int,
        default=1,
        help="Número de evoluciones para las preguntas."
    )
    
    # Argumentos del Módulo 2 (Reformulación)
    parser.add_argument(
        "--model_name_ref",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        #default="NousResearch/Meta-Llama-3-8B-Instruct",
        help="Nombre del modelo vLLM para reformulación."
    )
    parser.add_argument(
        "--base_url_ref",
        type=str,
        default="http://localhost:8000/v1/",
        help="URL base del servidor vLLM para reformulación."
    )
    parser.add_argument(
        "--umbral_similarity",
        type=float,
        default=0.75,
        help="Umbral mínimo de similitud semántica (0-1)."
    )
    parser.add_argument(
        "--umbral_rouge_max",
        type=float,
        default=0.5,
        help="Umbral máximo de similitud léxica (0-1)."
    )
    parser.add_argument(
        "--max_intentos_reformulacion",
        type=int,
        default=20,
        help="Número máximo de intentos para reformular una pregunta."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Tamaño del lote para procesamiento de reformulación."
    )
    
    # Argumentos del Módulo 3 (RAG)
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="paraphrase-multilingual:278m-mpnet-base-v2-fp16",
        #default="jina/jina-embeddings-v2-base-es:latest",
        help="Modelo de embeddings para RAG."
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        #default="llama3.1:8b",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        #default="NousResearch/Meta-Llama-3-8B-Instruct",
        help="Modelo LLM para RAG."
    )
    
    # Flags de control del pipeline
    parser.add_argument(
        "--skip_step1",
        action="store_true",
        help="Saltar el paso 1 (generación de preguntas)."
    )
    parser.add_argument(
        "--skip_step2",
        action="store_true",
        help="Saltar el paso 2 (reformulación de preguntas)."
    )
    parser.add_argument(
        "--skip_step3",
        action="store_true",
        help="Saltar el paso 3 (generación de respuestas RAG)."
    )
    
    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Definir rutas de archivos intermedios
    dataset_1_path = output_dir / "a1_dataset.csv"
    dataset_2_path = output_dir / "a2_dataset_reformulado.csv"
    dataset_3_path = output_dir / "a3_dataset_reformulado_RAG.csv"
    
    print("🚀 Iniciando pipeline completo de RAG...")
    print(f"📂 Documentos fuente: {args.documents_dir}")
    print(f"📁 Directorio de salida: {args.output_dir}")
    
    # ==========================================================================
    # PASO 1: GENERAR PREGUNTAS SINTÉTICAS
    # ==========================================================================
    if not args.skip_step1:
        print("\n" + "="*60)
        print("📝 PASO 1: Generando preguntas sintéticas...")
        print("="*60)
        
        # Verificar si el archivo ya existe
        if dataset_1_path.exists():
            print(f"⚠️  El archivo {dataset_1_path} ya existe.")
            respuesta = input("¿Desea sobrescribirlo? (s/n): ").lower().strip()
            if respuesta != 's':
                print("📋 Usando archivo existente para el Paso 1.")
            else:
                # Generar nuevo dataset
                generador = GeneradorPreguntas(
                    model_name=args.model_name_gen,
                    base_url=args.base_url_gen,
                    max_goldens_per_context=args.max_goldens_per_context,
                    num_evolutions=args.num_evolutions
                )
                
                print(f"🔄 Generando dataset desde: {args.documents_dir}")
                generador.generar_dataset(args.documents_dir, str(dataset_1_path))
                print(f"✅ Dataset generado: {dataset_1_path}")
        else:
            # Generar nuevo dataset
            generador = GeneradorPreguntas(
                model_name=args.model_name_gen,
                base_url=args.base_url_gen,
                max_goldens_per_context=args.max_goldens_per_context,
                num_evolutions=args.num_evolutions
            )
            
            print(f"🔄 Generando dataset desde: {args.documents_dir}")
            generador.generar_dataset(args.documents_dir, str(dataset_1_path))
            print(f"✅ Dataset generado: {dataset_1_path}")
    else:
        print("\n⏭️  Saltando Paso 1 (generación de preguntas)")
        if not dataset_1_path.exists():
            raise FileNotFoundError(f"Archivo requerido no encontrado: {dataset_1_path}")
    
    # ==========================================================================
    # PASO 2: REFORMULAR PREGUNTAS
    # ==========================================================================
    if not args.skip_step2:
        print("\n" + "="*60)
        print("🔄 PASO 2: Reformulando preguntas...")
        print("="*60)
        
        # Verificar que existe el archivo de entrada
        if not dataset_1_path.exists():
            raise FileNotFoundError(f"Archivo de entrada no encontrado: {dataset_1_path}")
        
        # Verificar si el archivo de salida ya existe
        if dataset_2_path.exists():
            print(f"⚠️  El archivo {dataset_2_path} ya existe.")
            respuesta = input("¿Desea sobrescribirlo? (s/n): ").lower().strip()
            if respuesta != 's':
                print("📋 Usando archivo existente para el Paso 2.")
            else:
                # Reformular preguntas
                reformulador = ReformuladorPreguntas(
                    model_name=args.model_name_ref,
                    base_url=args.base_url_ref,
                    umbral_similarity=args.umbral_similarity,
                    umbral_rouge_max=args.umbral_rouge_max,
                    max_intentos_reformulacion=args.max_intentos_reformulacion,
                    batch_size=args.batch_size
                )
                
                print(f"🔄 Reformulando preguntas desde: {dataset_1_path}")
                reformulador.procesar_dataset(str(dataset_1_path), str(dataset_2_path))
                print(f"✅ Preguntas reformuladas: {dataset_2_path}")
        else:
            # Reformular preguntas
            reformulador = ReformuladorPreguntas(
                model_name=args.model_name_ref,
                base_url=args.base_url_ref,
                umbral_similarity=args.umbral_similarity,
                umbral_rouge_max=args.umbral_rouge_max,
                max_intentos_reformulacion=args.max_intentos_reformulacion,
                batch_size=args.batch_size
            )
            
            print(f"🔄 Reformulando preguntas desde: {dataset_1_path}")
            reformulador.procesar_dataset(str(dataset_1_path), str(dataset_2_path))
            print(f"✅ Preguntas reformuladas: {dataset_2_path}")
    else:
        print("\n⏭️  Saltando Paso 2 (reformulación de preguntas)")
        if not dataset_2_path.exists():
            raise FileNotFoundError(f"Archivo requerido no encontrado: {dataset_2_path}")

    # ==========================================================================
    # PASO 3: GENERAR RESPUESTAS CON RAG
    # ==========================================================================
    if not args.skip_step3:
        print("\n" + "="*60)
        print("🤖 PASO 3: Generando respuestas con RAG...")
        print("="*60)
        
        # Verificar que existe el archivo de entrada
        if not dataset_2_path.exists():
            raise FileNotFoundError(f"Archivo de entrada no encontrado: {dataset_2_path}")
        
        # Verificar si el archivo de salida ya existe
        if dataset_3_path.exists():
            print(f"⚠️  El archivo {dataset_3_path} ya existe.")
            respuesta = input("¿Desea sobrescribirlo? (s/n): ").lower().strip()
            if respuesta != 's':
                print("📋 Usando archivo existente para el Paso 3.")
            else:
                # Generar respuestas RAG
                generar_respuestas_rag(
                    csv_path=str(dataset_2_path),
                    documents_dir=args.documents_dir,
                    output_path=str(dataset_3_path),
                    embedding_model=args.embedding_model,
                    llm_model=args.llm_model
                )
        else:
            # Generar respuestas RAG
            generar_respuestas_rag(
                csv_path=str(dataset_2_path),
                documents_dir=args.documents_dir,
                output_path=str(dataset_3_path),
                embedding_model=args.embedding_model,
                llm_model=args.llm_model
            )
    else:
        print("\n⏭️  Saltando Paso 3 (generación de respuestas RAG)")
        if not dataset_3_path.exists():
            print(f"⚠️  Archivo final no encontrado: {dataset_3_path}")

    # ==========================================================================
    # RESUMEN FINAL
    # ==========================================================================
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETADO")
    print("="*60)
    print(f"📂 Documentos procesados: {args.documents_dir}")
    print(f"📁 Archivos generados en: {args.output_dir}")
    print()
    
    # Mostrar información de archivos generados
    archivos_info = [
        (dataset_1_path, "Dataset inicial con preguntas sintéticas"),
        (dataset_2_path, "Dataset con preguntas reformuladas"),
        (dataset_3_path, "Dataset final con respuestas RAG")
    ]
    
    for archivo, descripcion in archivos_info:
        if archivo.exists():
            # Leer info básica del archivo
            try:
                df = pd.read_csv(archivo)
                print(f"✅ {archivo.name}: {len(df)} filas - {descripcion}")
                print(f"   Columnas: {list(df.columns)}")
            except Exception as e:
                print(f"✅ {archivo.name}: Existe - {descripcion}")
                print(f"   (Error leyendo detalles: {e})")
        else:
            print(f"❌ {archivo.name}: No generado - {descripcion}")
    
    print(f"\n🔗 Dataset final disponible en: {dataset_3_path}")
    print("🚀 ¡Pipeline RAG completado exitosamente!")


if __name__ == "__main__":
    main()
    