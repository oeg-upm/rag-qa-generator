# rag-qa-generator

Proyecto para generar respuestas a preguntas utilizando ragas

## Requisitos Previos

- Python 3.11
- pip
- conda (para instalar `libmagic`)

## Instalación

### Crear un entorno virtual

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Instalar las dependencias
```bash
pip install -r requirements.txt
```

### Instalar paquetes adicionales
```bash
pip install --prefer-binary "unstructured[pdf]"
pip install --prefer-binary "unstructured[md]"
pip install "accelerate>=0.26.0"
```

### Instalar libmagic
```bash
conda install -c conda-forge libmagic -y
```

## Uso

Actualizar en main.py el path con el directorio con los documentos para genenerar las preguntas.

### Ejecutar
```bash
python3 main.py
```