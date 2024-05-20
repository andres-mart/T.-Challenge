# Reto Data Scientist

Este reto está orientado a resolver, de una forma aproximada, un proyecto similar a algunos de los desarrollados en Turing Challenge desde el punto de vista del departamento de ciencia de datos.

## Apartados

### [Apartado 1: Crear un chatbot](Apartado%201)
El objetivo es crear un chatbot que tenga las siguientes funcionalidades:
- Una interfaz, por ejemplo, la interfaz de chatbot de Gradio.
- Ingestar varios documentos PDF largos para usarlos como base de conocimiento de un RAG.
- Usar una base de datos vectorial a elección.
- Implementar una memoria dinámica que mantenga la conversación y que cuando esta pase de X tokens se resuma de forma automática.
- La implementación ha de estar basada en Langchain.
- Si se detecta una pregunta que necesite de exactitud en la respuesta, el modelo ha de ser capaz de implementar y ejecutar código Python.

### [Apartado 2: Respuesta teórica](Apartado%202)
Dar respuesta a los siguientes puntos de forma teórica, sin necesidad de desarrollarlos, que guardan relación con las tecnologías utilizadas en el primer apartado:
- Diferencias entre 'completion' y 'chat' models.
- ¿Cómo forzar a que el chatbot responda 'sí' o 'no'? ¿Cómo parsear la salida para que siga un formato determinado?
- Ventajas e inconvenientes de RAG vs fine-tuning.
- ¿Cómo evaluar el desempeño de un bot de Q&A? ¿Cómo evaluar el desempeño de un RAG?

### [Apartado 3 (Opcional): Servicio local para detección de objetos](Apartado%203)
El objetivo es disponer de un servicio que tenga como entrada una imagen y que como salida proporcione un JSON con detecciones de coches y personas. Los puntos a cumplir son:
- No hay necesidad de entrenar un modelo. Se pueden usar preentrenados.
- El servicio ha de estar conteinerizado, es decir, una imagen Docker que al arrancar exponga el servicio.
- La petición al servicio se puede hacer desde Postman o herramienta similar o desde código Python.
- La solución ha de estar implementada en Python.

### [Apartado 4 (Opcional): Pasos necesarios para entrenar un modelo de detección](Apartado%204)
Además, plantear cuáles serían los pasos necesarios para entrenar un modelo de detección con categorías no existentes en los modelos preentrenados. Los puntos a centrar la explicación son:
- Pasos necesarios a seguir.
- Descripción de posibles problemas que puedan surgir y medidas para reducir el riesgo.
- Estimación de cantidad de datos necesarios así como de los resultados, métricas, esperadas.
- Enumeración y pequeña descripción (2-3 frases) de técnicas que se pueden utilizar para mejorar el desempeño, las métricas del modelo en tiempo de entrenamiento y las métricas del modelo en tiempo de inferencia.

## Enlaces a las carpetas de soluciones

- [Solución del Apartado 1](Apartado%201)
- [Solución del Apartado 3](Apartado%203)
