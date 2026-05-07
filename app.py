import streamlit as st
import pandas as pd
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# --- Configuration & Initialization ---
st.set_page_config(page_title="Internal Talent Radar", layout="wide")


# Ensure API key is set
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Initialize Gemini Models
@st.cache_resource
def load_models():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    return embeddings, llm

embeddings, llm = load_models()

# --- Task 3: AI Analysis Engine (Vector Store Setup) ---
@st.cache_resource
def setup_knowledge_base():
    # If no data exists yet, return None so the app doesn't crash
    if not os.path.exists("dummy_data.csv"):
        return None
        
    df = pd.read_csv("dummy_data.csv")
    
    # Convert CSV rows into LangChain Documents
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=row["Work_Content"],
            metadata={"employee_name": row["Employee_Name"]}
        )
        documents.append(doc)
        
    # Create ChromaDB Vector Store
    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store

# --- Task 4 & 5: HR Dashboard and RAG Retrieval Logic ---
st.title("🎯 Internal Talent Radar (ITR) Dashboard")
st.markdown("Uncover hidden leadership potential using everyday work evidence.")
st.divider()

vector_store = setup_knowledge_base()

if vector_store is None:
    st.warning("No data found. Please create 'dummy_data.csv' with 'Employee_Name' and 'Work_Content' columns to proceed.")
else:
    # UI: Job Description Input
    job_description = st.text_area(
        "Enter Open Role Description / Requirements", 
        height=150, 
        placeholder="e.g., Looking for a leader who can navigate ambiguity, foster cross-functional collaboration, and guide teams through complex technical challenges without micromanaging."
    )
    
    if st.button("Scout Internal Talent", type="primary"):
        if job_description.strip():
            with st.spinner("Analyzing internal work patterns..."):
                # RAG Step 1: Retrieve top 3 relevant documents based on semantic similarity
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                results = retriever.invoke(job_description)
                
                if not results:
                    st.info("No matching talent profiles found.")
                else:
                    st.subheader("Top Internal Candidates")
                    
                    # Group results by employee
                    candidates = {}
                    for res in results:
                        name = res.metadata["employee_name"]
                        if name not in candidates:
                            candidates[name] = []
                        candidates[name].append(res.page_content)
                    
                    # RAG Step 2: Use LLM to generate evidence-based reasoning
                    for name, works in candidates.items():
                        combined_work = "\n".join(works)
                        prompt = f"""
                        You are an HR Analyst evaluating internal talent.
                        Job Description: {job_description}
                        
                        Employee Name: {name}
                        Employee's Recent Work Highlights:
                        {combined_work}
                        
                        Write a 3-sentence summary explaining exactly why this employee's specific work history makes them a strong fit for this leadership role. Focus on evidence over buzzwords.
                        """
                        
                        evaluation = llm.invoke(prompt).content
                        
                        # Display Results
                        with st.expander(f"👤 **{name}** - Recommended Candidate", expanded=True):
                            st.markdown(f"**AI Evidence Summary:**\n{evaluation}")
                            st.markdown("**Source Work Material:**")
                            st.caption(f"\"{works[0][:200]}...\"")
        else:
            st.error("Please enter a job description.")