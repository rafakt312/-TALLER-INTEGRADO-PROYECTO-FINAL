1. Contexto
Durante el curso, todos los estudiantes han trabajado con el mismo conjunto de datos MHealth y el mismo problema:

Problema central: Reconocimiento de actividad humana en series de tiempo, utilizando ventanas temporales construidas a partir del dataset MHealth.

En las etapas previas se abordó:

Obtención de datos MHealth.

Análisis exploratorio de los patrones de comportamiento normal y anómalo.

Modelado y evaluación de algoritmos para detección de anomalías.

El Proyecto Final corresponde a la fase de despliegue, donde el foco es integrar lo anterior en un sistema funcional: Back End + Front End + documentación de uso y despliegue.

2. Objetivo del proyecto final
Desarrollar y desplegar, de forma individual, una aplicación web completa para reconocimiento de actividad humana en MHealth que:

Integre un modelo de reconocimiento de actividad humana en series de tiempo (derivado de la etapa de modelado).

Exponga un Back End con API para realizar detecciones.

Ofrezca un Front End que consuma la API y permita a un usuario no técnico interactuar con el sistema.

Incluya instrucciones de despliegue y uso claras y reproducibles.

Documente el uso de IA generativa (si la hubo) mediante una carpeta de prompts reutilizables.

El énfasis está en el despliegue técnico y en la integración de ciencia de datos con desarrollo de software a lo largo del ciclo de vida del proyecto.

La explicabilidad/XAI es deseable pero no obligatoria. Si se incluye, se considerará positivamente, pero no es requisito para aprobar.

3. Estructura del Proyecto Final (Etapas internas)
Dentro del proyecto final se distinguen tres etapas:

Etapa 1 – Desarrollo y despliegue técnico (código + repositorio)
Implementación del Back End y del Front End.

Integración del modelo de reconocimiento de actividad humana.

Consumo de la API desde el Front End.

Preparación de instrucciones de despliegue y uso.

Organización del repositorio Git del proyecto.

Inclusión de una carpeta de prompts si se utilizó IA generativa.

El sistema debe permitir la ingesta de archivos con el mismo formato con que se alimenta el modelo original (archivos .log del conjunto de datos de MHealth)

Éste último punto plantea una dificultad técnica: la entrada de datos que tiene que recibir el Front End tiene que ser compatible con los datos originales del conjunto Mhealth. Y por lo tanto para poder consumir la API, es necesario aplicar todo el procesamiento previo a los datos para que sea compatible con el modelo predictivo. Por lo tanto es necesario generalizar el trabajo realizado en la etapa de pre-procesamiento de los datos en funciones para procesar nuevos casos a modo de Pipeline.





Etapa 2 – Informe escrito (documento técnico)
Informe individual, máximo 12 páginas sin anexos.

Se puede incluir material suplementario como anexos o archivos adjuntos (figuras más grandes, código, ejemplos ampliados, configuraciones, etc.).

Este informe corresponde formalmente a la Etapa 2 del proyecto según la planificación del curso.

Etapa 3 – Presentación y demostración
Presentación individual de 8 minutos, más 2 minutos de preguntas.

Debe incluir una demostración en vivo del sistema (live demo).

Esta presentación corresponde a la Etapa 3 del proyecto y es requisito para aprobar el proyecto (ver detalle en evaluación).

4. Requisitos técnicos
4.1. Back End
El estudiante puede elegir libremente el lenguaje y framework (por ejemplo: Python + FastAPI/Flask/Django, Node.js/Express, Java/Spring, etc.).

Debe existir una API claramente definida con, al menos:

Un endpoint de verificación, por ejemplo:

GET /health → responde un mensaje simple indicando que el servicio está en ejecución.

Un endpoint de reconocimiento de actividad humana, por ejemplo:

POST /detect (nombre a elección razonable):

Entrada: archivo .log  (por ejemplo, JSON con las características necesarias). En otras palabras, predecir los datos de un sujeto que no participó en el entrenamiento.

Salida: resultado de detección de anomalías (Categoría, y corresponde la actividad que está realizando el sujeto.), más cualquier información adicional que se estime útil.

El Back End debe cargar y utilizar el modelo entrenado previamente (o una versión refinada consistente con el mismo problema).

4.2. Front End
Tecnología de libre elección (HTML/CSS/JS “puro”, React, Vue, etc.).

Requisitos mínimos:

Debe consumir la API del Back End (no se acepta lógica puramente local sin interacción con el servicio).

Debe permitir:

Ingresar datos o parámetros necesarios para formar la ventana / caso a evaluar.

Enviar una petición al Back End.

Mostrar de forma clara el resultado de la detección (por ejemplo, mensaje “mostrando la actividad”, gráficos simples, etc.).

La interfaz debe estar pensada para un usuario no experto (etiquetas comprensibles, mensajes claros).

4.3. Despliegue e instrucciones de uso
El sistema debe ser ejecutado en un entorno reproducible (por ejemplo, en una máquina local).

Docker es altamente recomendado. Si se utiliza, se deben incluir los archivos correspondientes (Dockerfile, docker-compose.yml, etc.).

Si no se utiliza Docker, el informe y/o el README deben indicar:

Sistema operativo objetivo.

Versiones de lenguajes y dependencias claves.

Pasos concretos para instalar dependencias y ejecutar Back End y Front End.

Cómo probar rápidamente que el sistema está funcionando.

5. Repositorio Git (obligatorio)
El proyecto debe estar en un repositorio Git (por ejemplo GitHub, GitLab o similar).

El informe debe incluir el enlace al repositorio.

Se espera un repositorio mínimamente ordenado, con:

Código separado en carpetas lógicas (backend, frontend, etc.).

Archivo README con instrucciones básicas de ejecución.

Carpeta prompts/ u otro nombre razonable para los prompts usados (si aplica).

6. Carpeta de prompts (IA generativa)
Si se utilizaron herramientas como ChatGPT, Copilot u otras, se debe incluir una carpeta (por ejemplo prompts/) con:

Archivos de texto o markdown que contengan los prompts más relevantes utilizados.

Opcionalmente, una línea que indique brevemente para qué parte del proyecto se usó cada prompt.

7. Informe escrito (Etapa 2)
Máximo 12 páginas sin anexos.

El material suplementario (tablas, figuras grandes, pseudocódigo, ejemplos extensos, etc.) puede ir en anexos o archivos adjuntos y no cuenta en el límite de páginas.

El informe debe incluir, al menos:

Resumen del sistema y del problema de detección de anomalías.

Contexto (cómo se conecta con las etapas previas del curso).

Arquitectura del sistema (incluyendo un diagrama simple de Front End, Back End y modelo).

Descripción de tecnologías escogidas para Back End y Front End, con una breve justificación.

Descripción de la API (endpoints, formato de entrada/salida, ejemplos breves).

Despliegue e instrucciones de uso.

Resultados de pruebas básicas (casos de prueba razonables para mostrar que el sistema funciona extremo a extremo).

Reflexión sobre el rol del despliegue en el ciclo de vida de ciencia de datos y posibles mejoras futuras.

(Opcional) Breve mención de elementos de XAI, si se incorporaron.

8. Presentación y demo (Etapa 3)
Duración:

8 minutos de presentación + 2 minutos de preguntas.

La presentación debe incluir:

Breve explicación del problema y del contexto (reconocimiento de actividad humana en MHealth).

Descripción de la arquitectura general (Front End, Back End, modelo).

Tecnologías utilizadas y razones principales de la elección.

Demostración en vivo del sistema (live demo): se debe mostrar el sistema funcionando, desde la interacción del usuario hasta la respuesta.

Breve reflexión sobre dificultades técnicas y aprendizajes.

La presentación (con demo en vivo) es requisito para aprobar el proyecto final.
Si el estudiante no realiza la presentación, no aprueba el proyecto, independientemente de la nota del informe o del código. Eventualmente habrá un plazo para una presentación recuperativa.

9. Evaluación (propuesta de distribución)
La ponderación global puede organizarse de la siguiente forma:

30% – Despliegue Back End + API (calidad técnica, robustez básica, integración del modelo).

25% – Front End + consumo de la API (usabilidad básica, claridad de la interacción, integración correcta con el Back End).

25% – Informe e instrucciones de uso (claridad, estructura, diagrama de arquitectura, reproducibilidad).

10% – Carpeta de prompts y uso responsable de IA generativa (si aplica).

10% – Presentación y demo (claridad al explicar, manejo del tiempo, demostración en vivo).

Puedes ajustar estos porcentajes en la pauta oficial, pero esta distribución sigue lo que comentaste y enfatiza el despliegue técnico.
