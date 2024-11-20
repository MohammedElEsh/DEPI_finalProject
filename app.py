import streamlit as st
import pickle
import PyPDF2
import docx
import spacy

# Load the spacy model
nlp = spacy.load('en_core_web_sm')

model_path = r"C:\Users\Compu Market\Downloads\Depi_graduation_project\week three (app)\model.pkl"
vectorlize_file = r"C:\Users\Compu Market\Downloads\Depi_graduation_project\week three (app)\vectorizer.pkl"

# Load the model and vectorizer
with open(model_path, 'rb') as model_file:
    logreg = pickle.load(model_file)

with open(vectorlize_file, 'rb') as vect_file:
    vect = pickle.load(vect_file)

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)  # Directly use uploaded_file
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to extract text from a Word file
def extract_text_from_word(uploaded_file):
    doc = docx.Document(uploaded_file)  # Directly use uploaded_file
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to analyze each sentence
def analyze_sentences(sentences):
    results = []
    for sentence in sentences:
        vectorized_sentence = vect.transform([sentence])
        prediction = logreg.predict(vectorized_sentence)
        results.append((sentence, prediction[0]))
    return results

# User interface
st.title("Sentiment Analysis App")
st.write("Enter a sentence for sentiment analysis or upload a PDF or Word file:")

# Option to input a sentence
sentence_input = st.text_area("Enter your sentence here:")
if st.button("Analyze Sentence"):
    cleaned_sentence = sentence_input  # You can add functions to clean the text if needed
    vectorized_sentence = vect.transform([cleaned_sentence])
    prediction = logreg.predict(vectorized_sentence)
    st.write("Sentiment analysis result:", prediction[0])

# Option to upload a PDF file
uploaded_pdf = st.file_uploader("Or upload a PDF file", type="pdf")
if uploaded_pdf is not None:
    document_text = extract_text_from_pdf(uploaded_pdf)  # Pass UploadedFile directly
    cleaned_text = document_text  # You can add functions to clean the text if needed
    doc = nlp(cleaned_text)  # Process the text with spacy
    sentences = [sent.text for sent in doc.sents]  # Split text into sentences using spacy
    analysis_results = analyze_sentences(sentences)  # Analyze each sentence
    for sentence, sentiment in analysis_results:
        st.write(f"Sentence: '{sentence}' → Sentiment Analysis Result: {sentiment}")

# Option to upload a Word file
uploaded_word = st.file_uploader("Or upload a Word file", type="docx")
if uploaded_word is not None:
    document_text = extract_text_from_word(uploaded_word)  # Pass UploadedFile directly
    cleaned_text = document_text  # You can add functions to clean the text if needed
    doc = nlp(cleaned_text)  # Process the text with spacy
    sentences = [sent.text for sent in doc.sents]  # Split text into sentences using spacy
    analysis_results = analyze_sentences(sentences)  # Analyze each sentence
    for sentence, sentiment in analysis_results:
        st.write(f"Sentence: '{sentence}' → Sentiment Analysis Result: {sentiment}")
