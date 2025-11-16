# La cocina de los LLM
by Jose Bengoa, 16/11/2025.
Con la ayuda de Anthropic (Claude) y OpenAI (chatGPT)

Este script contiene una implementación básica, a modo de ejemplo, de vectorización semántica aplicada a entidades biológicas (Biodiversity Vectorization, BV). Es una técnica que se propone para aplicar los avances del Natural Language Processing (NLP) al estudio espacial y funcional de la biodiversidad.

Mas detalles en: Bengoa, J. & Turkina, S. 2025. La cocina de los grandes modelos del lenguaje aplicada al procesamiento de información sobre biodiversidad: exprimiendo GBIF. Jornadas sobre Información de Biodiversidad y Administraciones Ambientales 2025 - Zaragoza.

Sigue un planteamiento equivalente al embedding que se hace en el procesamiento del lenguaje natural para convertir las bases de datos de biodiversidad en bases de datos de vectores.

Los vectores se obtienen mediante un modelo entrenado con la información que aporta la coocurrencia de especies en unas mismas localidades, celdas, etc.
Esta vectorización facilita el procesado de la biodiversidad, para tareas como:
* Detectar citas anómalas
* Predecir hábitats
* Rellenar huecos en zonas poco muestreadas
* Generar un mapa de similitud entre celdas
* Generar un mapa de similitud para una especie concreta

Para ello se genera un corpus de especies que se descarga de la base de datos de biodiversidad GBIF (www.gbif.org) usando su API (https://techdocs.gbif.org/en/openapi/, https://github.com/gbif/gbif-api) usando tanto el paquete pygbif (https://github.com/gbif/pygbif), como las llamadas a la API mediante HTTP.


## **Descripción del script**

Este script, desarrollado para las *Jornadas sobre Información de Biodiversidad y Administraciones Ambientales 2025*, muestra paso a paso cómo transformar datos de biodiversidad procedentes de GBIF en representaciones vectoriales que permiten aplicar técnicas modernas de análisis semántico y modelos de aprendizaje automático.

El objetivo principal es ilustrar cómo pasar de datos de ocurrencias —registros puntuales de especies— a **embeddings**: vectores numéricos que capturan relaciones ecológicas, coocurrencias y similitudes entre especies y entre celdas espaciales.


### **Qué hace el script**

1. **Instala las dependencias necesarias**

   * `pygbif` para descargar datos desde la API de GBIF
   * `gensim` para generar embeddings con Word2Vec
   * `scikit-learn` para calcular similitud y otras tareas
   * `pandas`, `numpy`, `matplotlib`, `requests`, para diversas tareas

2. **Importa las librerías y configura parámetros básicos**
   Se definen el reino biológico, el país, el tamaño de página para las descargas y otros parámetros de control.

3. **Descarga masiva de ocurrencias desde GBIF**
   Utilizando `pygbif` o la API REST:

   * Se obtiene un corpus de especies y coordenadas
   * Se filtra por calidad de datos (coordenadas válidas, sin flags críticos)
   * Se agregan los datos por celdas (rejilla espacial regular)

4. **Construcción del corpus ecológico para entrenar el modelo**
   Cada celda se convierte en una *"frase"* y las especies presentes en *"palabras"*.
   Esto permite aplicar técnicas de Procesamiento del Lenguaje Natural a la biodiversidad.

   Ejemplo:

   ```
   Celda A → ["Quercus ilex", "Pinus halepensis", "Thymus mastichina"]
   Celda B → ["Pinus sylvestris", "Juniperus communis"]
   ```

5. **Entrenamiento del modelo Word2Vec**
   Con `gensim`, se entrena un modelo que aprende:

   * semejanza ecológica entre especies
   * patrones de coocurrencia
   * afinidad ambiental entre celdas
   * gradientes biogeográficos

6. **Generación de embeddings**
   El script calcula:

   * **embedding de especies**: vector que representa el "nicho semántico" basado en coocurrencias
   * **embedding de celdas**: media de los vectores de las especies presentes en cada celda

7. **Normalización y reducción de dimensionalidad (PCA)**
   Para visualizar los resultados, se aplica PCA y se generan gráficos donde:

   * las especies se agrupan según afinidad ecológica
   * las celdas forman clústeres ambientales o biogeográficos

8. **Aplicaciones prácticas**
   El script incluye ejemplos de:

   * **Detección de anomalías**: celdas que no encajan con su entorno
   * **Predicción de hábitats esperables**: similitud celda–especie
   * **Mapas de similitud**: visualización de patrones espaciales complejos
   * **[Relleno de huecos**: identificar áreas poco muestreadas con alta afinidad]

9. **Caso práctico final**
   Una sección específica permite seleccionar una especie concreta y generar:

   * su vector de hábitat (embedding medio de celdas donde aparece)
   * un mapa coloreado según la similitud ecológica con ese vector
     
   Esto sirve para detectar:
   * hábitats favorables no muestreados
   * posibles errores o citas anómalas
   * áreas biogeográficamente interesantes

---

## **Parámetros Configurables**

El script permite ajustar múltiples parámetros según las necesidades del análisis:

### Parámetros de descarga de GBIF
* `KINGDOM`: Reino taxonómico a estudiar (ej. "Animalia", "Plantae")
* `COUNTRY`: Código ISO del país (ej. "ES" para España)
* `PAGE_SIZE`: Número de registros por página (máx. 300)
* `FETCH_RECORDS`: Número total de registros a descargar
* `BOUNDING_BOX`: Coordenadas geográficas del área de estudio

### Parámetros del modelo Word2Vec
* `VECTOR_SIZE`: Dimensión de los vectores (típicamente 50-300)
* `WINDOW`: Tamaño de la ventana de contexto
* `MIN_COUNT`: Frecuencia mínima de aparición de una especie
* `WORKERS`: Número de hilos para procesamiento paralelo
* `EPOCHS`: Número de iteraciones de entrenamiento

### Parámetros espaciales
* `CELL_SIZE`: Tamaño de la celda en grados (determina la resolución espacial)
* `MIN_SPECIES_PER_CELL`: Número mínimo de especies por celda para incluirla en el análisis

---

## **Contexto Técnico y Métrica de Similitud**

El script implementa la comparación entre especies o celdas utilizando una métrica común en el análisis de vectores: la **Similitud del Coseno**.

### 1. Vectores de Celdas (Embeddings)

Cada celda geográfica o hábitat se representa como un vector de alta dimensión (un embedding), donde cada dimensión codifica información semántica sobre la biodiversidad de esa celda. Estos vectores se almacenan en el diccionario `cell_embeddings`.

### 2. Similitud del Coseno

La **Similitud del Coseno** (Cosine Similarity) mide el ángulo entre dos vectores en un espacio multidimensional, independientemente de su magnitud.

* Un valor de **1** indica que los vectores son idénticos en dirección (máxima similitud)
* Un valor de **0** indica que los vectores son ortogonales (sin relación)
* Un valor de **-1** indica que los vectores son opuestos

En el script, la similitud entre un vector de hábitat (`habitat_vec`) y el vector de una celda específica (`vec`) se calcula usando la función `cosine_similarity` de scikit-learn. 

La fórmula matemática para la Similitud del Coseno entre dos vectores **A** y **B** es:

```
Similitud(A,B) = (A · B) / (||A|| × ||B||)
```

Donde:
* **A · B** es el producto escalar (dot product)
* **||A||** y **||B||** son las magnitudes (norma euclidiana) de los vectores

#### Interpretación en el contexto ecológico

* **Similitud alta (cercana a 1)**: Las celdas comparten composiciones de especies similares, sugiriendo condiciones ambientales o biogeográficas parecidas
* **Similitud baja (cercana a 0)**: Las celdas tienen comunidades biológicas muy diferentes
* **Similitud negativa**: Raramente ocurre en este contexto, pero indicaría composiciones completamente opuestas

### 3. Manejo de Celdas sin Vectorización

Para las celdas identificadas (`cid`) que no están presentes en la base de datos de vectores (`cell_embeddings`), el script asigna un vector de ceros del tamaño `VECTOR_SIZE`, asegurando que el cálculo de la similitud se pueda realizar con un valor por defecto:

```python
vec = cell_embeddings.get(cid, np.zeros((1, VECTOR_SIZE)))
```

Esto resulta en una similitud baja o neutra para celdas sin datos, evitando errores en el análisis y permitiendo identificar áreas con información insuficiente.

El resultado de estas comparaciones se almacena en una columna llamada `sim_species` dentro del DataFrame `cell_df`, permitiendo mapear la similitud de cada celda respecto al hábitat de referencia mediante visualizaciones espaciales coloreadas.

---

## **Flujo de Trabajo Completo**

```
DATOS BRUTOS DE GBIF
         ↓
LIMPIEZA Y FILTRADO
         ↓
AGREGACIÓN ESPACIAL (celdas)
         ↓
CONSTRUCCIÓN DEL CORPUS
         ↓
ENTRENAMIENTO Word2Vec
         ↓
GENERACIÓN DE EMBEDDINGS
         ↓
CÁLCULO DE SIMILITUDES
         ↓
VISUALIZACIÓN Y ANÁLISIS
```

---

## **Requisitos del Sistema**

* Python 3.7 o superior
* Conexión a internet para descargar datos de GBIF
* Memoria RAM: mínimo 4GB recomendado para conjuntos de datos grandes
* Espacio en disco: variable según el volumen de datos descargados

---

## **Salidas y Resultados**

El script genera:

1. **Archivos CSV** con los datos descargados y procesados
2. **Gráficos de dispersión PCA** mostrando la estructura de los embeddings
3. **Mapas de calor** con similitudes ecológicas
4. **Modelos Word2Vec entrenados** (guardables para uso posterior)
5. **Matrices de similitud** entre especies y celdas

---

## **En conjunto**

El cuaderno demuestra cómo un flujo de trabajo relativamente sencillo permite pasar de los datos brutos de GBIF a herramientas avanzadas basadas en vectorización semántica, facilitando nuevas posibilidades de análisis ecológico y gestión ambiental.

Esta aproximación abre la puerta a:
* Análisis predictivos de distribución de especies
* Identificación de áreas prioritarias para conservación
* Detección de sesgos en el muestreo
* Modelización de nicho ecológico basada en similitud semántica
* Evaluación de conectividad biogeográfica

Todo ello utilizando técnicas probadas en el campo del procesamiento del lenguaje natural, adaptadas inteligentemente al dominio de la biodiversidad.
