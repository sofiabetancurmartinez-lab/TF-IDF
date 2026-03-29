import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.title("🔍 Demo TF-IDF en Español")

# Documentos de ejemplo más generales
default_docs = """La universidad organiza un evento cultural el próximo viernes.
Los estudiantes presentarán proyectos interactivos en la feria académica.
El laboratorio de diseño abre sus puertas desde las ocho de la mañana.
La biblioteca ofrece espacios tranquilos para estudiar en grupo.
El concierto principal será en la plazoleta central al final de la tarde.
Los visitantes podrán inscribirse en talleres creativos durante la jornada."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def generar_preguntas_sugeridas(documents):
    sugerencias = []
    for doc in documents[:5]:
        sugerencias.append(f"¿De qué trata: '{doc[:40]}...' ?")
    return sugerencias

# Inicializar estado
if "question" not in st.session_state:
    st.session_state.question = ""

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos (uno por línea):", default_docs, height=180)
    question = st.text_input(
        "❓ Escribe tu pregunta:",
        value=st.session_state.question if st.session_state.question else "¿Dónde será el concierto principal?"
    )

documents_preview = [d.strip() for d in text_input.split("\n") if d.strip()]
preguntas_sugeridas = generar_preguntas_sugeridas(documents_preview)

with col2:
    st.markdown("### 💡 Preguntas sugeridas")
    for i, sugerencia in enumerate(preguntas_sugeridas):
        if st.button(sugerencia, key=f"sugerencia_{i}", use_container_width=True):
            st.session_state.question = sugerencia
            st.rerun()

# Actualizar pregunta desde session_state
if st.session_state.question:
    question = st.session_state.question

if st.button("🔍 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )

        X = vectorizer.fit_transform(documents)

        st.markdown("### 📊 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)

        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.markdown("### 🎯 Resultado")
        st.markdown(f"**Tu pregunta:** {question}")

        # Mostrar ranking completo
        resultados = pd.DataFrame({
            "Documento": documents,
            "Similitud": similarities
        }).sort_values(by="Similitud", ascending=False)

        st.markdown("### 📌 Ranking de similitud")
        st.dataframe(resultados.round(3), use_container_width=True)

        if best_score > 0.01:
            st.success(f"**Frase más relacionada:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")
        else:
            st.warning(f"**Resultado con baja confianza:** {best_doc}")
            st.info(f"📉 Similitud: {best_score:.3f}")