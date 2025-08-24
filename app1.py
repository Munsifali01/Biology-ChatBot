import streamlit as st
import pandas as pd
import random

# =============================
# 1. Data (Directly Inside Code)
# =============================
data = [
    {"class": "FSc Part 1", "chapter": "Cell Biology", "question": "What is the basic unit of life?", "answer": "Cell"},
    {"class": "FSc Part 1", "chapter": "Biomolecules", "question": "Which biomolecule stores genetic information?", "answer": "DNA"},
    {"class": "FSc Part 1", "chapter": "Enzymes", "question": "Enzymes act as?", "answer": "Catalysts"},
    {"class": "FSc Part 2", "chapter": "Human Physiology", "question": "Which organ pumps blood?", "answer": "Heart"},
    {"class": "FSc Part 2", "chapter": "Human Physiology", "question": "What is the functional unit of kidney?", "answer": "Nephron"},
    {"class": "FSc Part 2", "chapter": "Genetics", "question": "Who is called the father of genetics?", "answer": "Gregor Mendel"},
    {"class": "MDCAT", "chapter": "Cell Division", "question": "Mitosis results in how many daughter cells?", "answer": "2"},
    {"class": "MDCAT", "chapter": "Cell Division", "question": "Meiosis results in how many daughter cells?", "answer": "4"},
    {"class": "MDCAT", "chapter": "Human Blood", "question": "What is the universal donor blood group?", "answer": "O-"},
    {"class": "MDCAT", "chapter": "Human Blood", "question": "What is the universal recipient blood group?", "answer": "AB+"},
    # ---- آپ یہاں مزید MCQs add کر سکتے ہیں اسی pattern پر ----
]

# Convert list into DataFrame
df = pd.DataFrame(data)

# =============================
# 2. Streamlit App UI
# =============================
st.title("📘 Biology MCQs Quiz App")

# Select Class & Chapter
selected_class = st.selectbox("Select Class", df["class"].unique())
filtered_df = df[df["class"] == selected_class]

selected_chapter = st.selectbox("Select Chapter", filtered_df["chapter"].unique())
chapter_df = filtered_df[filtered_df["chapter"] == selected_chapter]

# Pick a random question
if st.button("Next Question"):
    q = random.choice(chapter_df.to_dict(orient="records"))
    st.subheader("❓ " + q["question"])
    
    if st.button("Show Answer"):
        st.success("✅ Answer: " + q["answer"])
