# NPTEL Preparation Buddy - Generative AI Study Assistant  

An AI-powered Generative AI Study Assistant built with Python, LangChain, Google Gemini API, and ChromaDB.

This project helps students upload NPTEL lecture PDFs, interact with them via contextual Q&A, get concise summaries, and even generate quiz-style MCQs for exam preparation.

Features:

1.Multi-PDF Q&A → Ask natural language questions across multiple NPTEL PDFs.

2.Summarization → Generate concise, revision-ready summaries of lecture materials.

3.Quiz Generator → Create MCQ-style quiz questions with correct answers for self-testing.

4.Contextual Memory → Maintains chat history for better, more relevant responses.

5.User-Friendly Interface → Built with Streamlit for an interactive and simple web UI.

Tech Stack:

1.Python

2.LangChain (document parsing, chaining)

3.Google Gemini API (embeddings + conversational model)

4.ChromaDB (vector storage & retrieval)

5.Streamlit (frontend interface)

6.PyPDF (PDF parsing)

Usage:

1.Upload one or more NPTEL PDFs (lecture notes, assignments, textbooks).

2.Use the Q&A tab to ask contextual questions.

3.Generate summaries for quick revision.

4.Create quiz questions to practice for exams.

Why This Project?

1.Unlike generic GenAI chatbots, NPTEL Prep Buddy is a personalized academic assistant designed to enhance learning and boost exam preparation efficiency.

2.It demonstrates real-world applications of LangChain, vector databases, and LLM integration in the education domain

Installation & Setup

1. Clone the repository:

   ---
   git clone https://github.com/Rebeccacharles/NPTEL_Prep_Buddy/tree/main

   cd NPTEL-Prep-Buddy
   ---

3. Install dependencies:

   
   pip install -r requirements.txt
   

4. Set up Google Gemini API Key:

   * Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

    * Export it in your environment:

    (USE YOUR GEMINI API KEY)

6. Run the app:

   ---
   streamlit run app.py
   ---


