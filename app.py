from flask import Flask, request, jsonify
import base64
import json
import os
import fitz
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from flask_cors import CORS
import google.generativeai as genai
import numpy as np
import faiss
from typing_extensions import TypedDict
from grade import grade_pipeline
import random
import traceback
import re


# Define the schema for the JSON response
class PromptResponse(TypedDict):
    prompt: str

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")


def _load_api_key_from_env_files():
    """Try to read Gemini API key from local .env files if env vars are missing."""
    env_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
    ]

    for env_path in env_paths:
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as env_file:
                for raw_line in env_file:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key in ("GOOGLE_API_KEY", "GEMINI_API_KEY") and value:
                        return value
        except OSError:
            continue

    return None


def configure_genai_client():
    """Configure Gemini SDK once per request path and fail with clear guidance if key is missing."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or _load_api_key_from_env_files()
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in environment or .env file."
        )
    genai.configure(api_key=api_key)


def _embedding_model_candidates():
    """Return embedding model candidates in fallback order."""
    candidates = [
        EMBEDDING_MODEL,
        "models/text-embedding-004",
        "models/embedding-001",
    ]

    # Preserve order while removing duplicates.
    seen = set()
    unique_candidates = []
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            unique_candidates.append(model)
    return unique_candidates


def _embed_content_with_fallback(content):
    """Embed content while handling model-id differences across Gemini API versions."""
    last_error = None
    for model_name in _embedding_model_candidates():
        try:
            response = genai.embed_content(model=model_name, content=content)
            return response["embedding"]
        except Exception as exc:
            error_text = str(exc).lower()
            if "is not found" in error_text or "not supported for embedcontent" in error_text:
                last_error = exc
                continue
            raise

    raise RuntimeError(
        "No compatible embedding model found for this API key/project. "
        "Try setting GEMINI_EMBEDDING_MODEL=models/embedding-001 or models/text-embedding-004. "
        f"Last error: {last_error}"
    )


def _tokenize_for_retrieval(text):
    """Simple tokenizer for fallback retrieval when embeddings are unavailable."""
    if not text:
        return set()
    return set(re.findall(r"[\w\u0600-\u06FF]+", text.lower()))


def _keyword_retrieve_top_chunks(query, data, top_k=5):
    """Fallback retriever using token overlap; avoids failing the whole request."""
    query_tokens = _tokenize_for_retrieval(query)
    scored = []
    for chunk in data:
        chunk_text = chunk.get('chunk_text', '')
        chunk_tokens = _tokenize_for_retrieval(chunk_text)
        overlap = len(query_tokens & chunk_tokens)
        # Add tiny length-aware tie breaker so deterministic ordering is stable.
        score = overlap + (min(len(chunk_tokens), 300) * 1e-6)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in scored[:max(top_k, 1)]]
    return selected




def decode_base64_to_pdf(base64_string: str):
    """Decodes a Base64 string to a PDF file.

    Args:
        base64_string: The Base64 encoded string.
    Returns:
        True on success, False otherwise
    """
    try:

        pdf_bytes = base64.b64decode(base64_string.encode("utf-8")) #encode to bytes for decoding
       # Create a BytesIO object from the bytes
        pdf_stream = BytesIO(pdf_bytes)
        
        # Open PDF directly from memory
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")

        return pdf_document
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"An error occurred during decoding: {error_traceback}")
        return False
    

# Optional: Enhanced chunking with content-aware splitting
def process_pdf_with_content_awareness(pdf_base64, chunk_size=500, chunk_overlap=100):
    """
    Process PDF with content-aware chunking strategies.
    
    Args:
        pdf_base64 (str): Encoded PDF file
        chunk_size (int): Maximum characters per chunk
        chunk_overlap (int): Number of characters to overlap between chunks
    
    Returns:
        list: List of dictionaries containing text chunks and metadata
    """
    try:
        # Load PDF
        doc = decode_base64_to_pdf(pdf_base64)
        
        # Extract text and create documents in LangChain format
        documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                        "total_pages": len(doc)
                    }
                )
            )
        
        doc.close()
        # Create a content-aware text splitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                "。",    # Chinese/Japanese period
                ".",    # English period
                "！",   # Chinese/Japanese exclamation
                "!",    # English exclamation
                "？",   # Chinese/Japanese question mark
                "?",    # English question mark
                "；",   # Chinese/Japanese semicolon
                ";",    # English semicolon
                ",",    # English comma
                " ",    # Space
                ""      # Character
            ],
            is_separator_regex=False
        )
        
        # Split documents
        chunks = text_splitter.split_documents(documents)
        
        # Process chunks with enhanced metadata
        print("CAME HERE")
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'chunk_text': chunk.page_content,
                'page_num': chunk.metadata.get('page', None),
            })
        return processed_chunks
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in process_pdf_with_content_awareness: {error_traceback}")
        return []
    


async def RAG_pdfs(query, data, top_k=5):
    """
    Perform Retrieval-Augmented Generation (RAG) on a set of PDF chunks.
        
    Args:
        query (str): The query string to search for relevant chunks.
        data (list): List of dictionaries containing chunked text and metadata.
    
    Returns:
        list: List of dictionaries containing relevant chunks and their metadata.
    """
    
    if not data:
        return []
    texts = [chunk['chunk_text'] for chunk in data]
    configure_genai_client()

    try:
        embeddings = np.array(_embed_content_with_fallback(texts), dtype=np.float32)

        embedding_dim = embeddings.shape[1]
        data_index = faiss.IndexFlatL2(embedding_dim)
        data_index.add(embeddings)

        query_embedding = np.array(_embed_content_with_fallback([query]), dtype=np.float32)

        D, I = data_index.search(query_embedding, top_k)

        top_chunks = [data[i] for i in I[0]]
        return top_chunks
    except RuntimeError as emb_error:
        print(f"Embedding unavailable, using keyword fallback retrieval: {emb_error}")
        return _keyword_retrieve_top_chunks(query, data, top_k)

class QAPair(TypedDict):
    question: str
    answer: str

# def generate_questions(prompt, chunks):
#     """
#     Generate 3 question-answer pairs focused on a specific PDF section, using context from related chunks.
    
#     Args:
#         prompt (str): Target section from educational PDF
#         chunks (list): Relevant context chunks from RAG system
    
#     Returns:
#         list: Dicts with 'question' and 'answer' keys
#     """
    
#     context = "\n".join([chunk['chunk_text'] for chunk in chunks])
    
#     instruction = (
#         "As an educational content creator, generate 3 precise question-answer pairs focusing on "
#         "the key concepts in the 'Target PDF Section' below. Use these guidelines:\n"
#         "1. Base questions primarily on the Target Section\n"
#         "2. Use Context only to enhance answers with related information\n"
#         "3. Questions should test understanding of main concepts\n"
#         "4. Answers must be concise and factually accurate\n"
#         "5. Format as JSON list of dictionaries, each containing 'question' and 'answer'"
#         "6. Use the 'Target PDF Section' as the main focus for questions and dont repate the same question"
#         "7. dont ask questions like , as per the text , or according to the text , or as mentioned in table 7 etc , becasue the students cant see the text or table"
#         "8. you can ask questions with long answers like explain the consept of , or what is the importance of , or what is the difference between , etc"
#     )
    
#     combined_prompt = (
#         f"{instruction}\n\n"
#         f"Context:\n{context}\n\n"
#         f"Target PDF Section:\n{prompt}\n\n"
#         "JSON response:"
#     )
#     model = genai.GenerativeModel('gemini-2.0-flash')
#     result = model.generate_content(
#         combined_prompt,
#         generation_config=genai.GenerationConfig(
#             temperature=0.3,
#             response_mime_type="application/json",
#             # response_schema=list[QAPair]  # Corrected schema
#         )
#     )
#     return json.loads(result.candidates[0].content.parts[0].text) if isinstance(json.loads(result.candidates[0].content.parts[0].text), list) else []


def detect_language(text):
    """
    Detect if the text is primarily Arabic or English.
    
    Args:
        text (str): Text to analyze
    
    Returns:
        str: 'ar' for Arabic, 'en' for English
    """
    # Count Arabic characters (Unicode range for Arabic script)
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
    # Count English/Latin characters
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # If more than 30% of characters are Arabic, consider it Arabic text
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    if total_chars > 0 and arabic_chars / total_chars > 0.3:
        return 'ar'
    else:
        return 'en'


def generate_questions(prompt, chunks):
    """
    Generate 3 question-answer pairs focused on a specific PDF section, using context from related chunks.
    Questions will be generated in the same language as the input text.
    
    Args:
        prompt (str): Target section from educational PDF
        chunks (list): Relevant context chunks from RAG system
    
    Returns:
        list: Dicts with 'question' and 'answer' keys
    """
    
    context = "\n".join([chunk['chunk_text'] for chunk in chunks])
    
    # Detect the language of the input text
    language = detect_language(prompt)
    
    if language == 'ar':
        # Arabic instruction
        improved_instruction = (
            "بصفتك منشئ محتوى تعليمي، مهمتك هي إنشاء ثلاثة أزواج من الأسئلة والأجوبة المتميزة التي تركز على المفاهيم الأساسية المعروضة في 'القسم المستهدف من PDF' المقدم أدناه. اتبع هذه الخطوات لضمان فعالية الأسئلة للتعلم:\n\n"
            "1. *تحديد المفاهيم الأساسية*: اقرأ القسم المستهدف من PDF بعناية وحدد المفاهيم أو الأفكار أو العمليات الثلاثة الأكثر أهمية التي يعرضها.\n\n"
            "2. *إنشاء الأسئلة*: لكل من المفاهيم الثلاثة، أنشئ سؤالاً يختبر فهم الطالب لذلك المفهوم. يجب أن تكون الأسئلة:\n"
            "   - مبنية بشكل أساسي على القسم المستهدف من PDF.\n"
            "   - مستقلة ولا تشير إلى النص مباشرة (مثل تجنب 'حسب النص' أو 'وفقاً للشكل 1').\n"
            "   - متنوعة في النوع، مثل السؤال عن التعريفات أو التفسيرات أو المقارنات أو التطبيقات.\n\n"
            "3. *استخدام السياق للإجابات*: يتكون 'السياق' المقدم من أقسام ذات صلة من PDF. استخدم هذا السياق لإثراء إجاباتك من خلال تضمين تفاصيل أو أمثلة أو اتصالات ذات صلة غير موجودة في القسم المستهدف. ومع ذلك، تأكد من أن جوهر الإجابة يستند إلى القسم المستهدف.\n\n"
            "4. *ضمان الوضوح والدقة*: تأكد من أن كل سؤال واضح وأن الإجابة موجزة ودقيقة من الناحية الواقعية وتتناول السؤال مباشرة.\n\n"
            "5. *تجنب التكرار*: يجب أن يغطي كل سؤال مفهوماً أو جانباً مختلفاً من القسم المستهدف.\n\n"
            "6. *التنسيق*: قدم استجابتك كقائمة JSON من المعاجم، حيث يحتوي كل معجم على مفاتيح 'question' و 'answer'.\n\n"
            "تذكر، الهدف هو مساعدة الطلاب على تعزيز تعلمهم من خلال أسئلة مستهدفة تشجعهم على تذكر وتطبيق المادة. يجب أن تكون جميع الأسئلة والأجوبة باللغة العربية."
        )
        
        combined_prompt = (
            f"{improved_instruction}\n\n"
            f"القسم المستهدف من PDF:\n{prompt}\n\n"
            f"السياق:\n{context}\n\n"
            "أنشئ استجابة JSON:"
        )
    else:
        # English instruction (existing)
        improved_instruction = (
            "As an educational content creator, your task is to generate three distinct question-answer pairs that focus on the key concepts presented in the 'Target PDF Section' provided below. Follow these steps to ensure the questions are effective for learning:\n\n"
            "1. *Identify Key Concepts*: Read the Target PDF Section carefully and identify the three most important concepts, ideas, or processes it presents.\n\n"
            "2. *Generate Questions*: For each of the three concepts, create a question that tests the student's understanding of that concept. The questions should:\n"
            "   - Be based primarily on the Target PDF Section.\n"
            "   - Be standalone and not reference the text directly (e.g., avoid 'as per the text' or 'according to Figure 1').\n"
            "   - Vary in type, such as asking for definitions, explanations, comparisons, or applications.\n\n"
            "3. *Use Context for Answers*: The 'Context' provided consists of related sections from the PDF. Use this context to enrich your answers by including relevant details, examples, or connections that are not present in the Target Section. However, ensure that the core of the answer is based on the Target Section.\n\n"
            "4. *Ensure Clarity and Accuracy*: Make sure that each question is clear and that the answer is concise, factually accurate, and directly addresses the question.\n\n"
            "5. *Avoid Repetition*: Each question should cover a different concept or aspect of the Target Section.\n\n"
            "6. *Format*: Present your response as a JSON list of dictionaries, where each dictionary contains 'question' and 'answer' keys.\n\n"
            "Remember, the goal is to help students reinforce their learning through targeted questions that encourage them to recall and apply the material. All questions and answers should be in English."
        )
        
        combined_prompt = (
            f"{improved_instruction}\n\n"
            f"Target PDF Section:\n{prompt}\n\n"
            f"Context:\n{context}\n\n"
            "Generate the JSON response:"
        )
    
    configure_genai_client()
    model = genai.GenerativeModel(GENERATION_MODEL)
    result = model.generate_content(
        combined_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.5,  # Increased for more creative and varied questions
            response_mime_type="application/json"
        )
    )
    
    try:
        response_text = result.candidates[0].content.parts[0].text
        qa_pairs = json.loads(response_text)
        if isinstance(qa_pairs, list):
            return qa_pairs
        else:
            return []  # Return empty list if response is not a list
    except (json.JSONDecodeError, KeyError, IndexError):
        return []  # Return empty list if parsing fails

# Note: Ensure that the model and API are correctly configured and that the necessary imports are in place.
@app.route('/process', methods=['post'])
async def process():
    try:
        print("Processing PDF")
        data = request.get_json()
        pdf = data.get('pdf')
        paragrpath = data.get('paragrpath')
        if not pdf or not paragrpath:
            return jsonify({'error': 'Missing parameters'}), 400
        chunks = process_pdf_with_content_awareness(pdf)
        result = await RAG_pdfs(paragrpath, chunks, 3)
        questions = generate_questions(paragrpath, result)
        return jsonify(questions), 200
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in /process: {error_traceback}")
        return jsonify({'error': str(e), 'traceback': error_traceback}), 500
    



@app.route('/generate_full', methods=['POST'])
async def generate_full():
    try:
        print("Processing PDF")
        data = request.get_json()
        pdfs = data.get('pdfs')
        if not pdfs:
            return jsonify({'error': 'Missing pdfs'}), 400
        
        # Extract chunks from all PDFs
        chunks = []
        for pdf in pdfs:
            pdf_chunks = process_pdf_with_content_awareness(pdf['data'])
            for chunk in pdf_chunks:
                chunk['source'] = pdf.get('name', 'unknown')
            chunks.extend(pdf_chunks)
        
        if not chunks:
            return jsonify({'error': 'No chunks extracted'}), 400
        
        # Compute embeddings for all chunks once
        texts = [chunk['chunk_text'] for chunk in chunks]
        configure_genai_client()
        embeddings = np.array(_embed_content_with_fallback(texts), dtype=np.float32)
        
        # Build FAISS index once
        data_index = faiss.IndexFlatL2(embeddings.shape[1])
        data_index.add(embeddings)
        
        # Sample 10 random chunks (or fewer if not enough chunks)
        num_samples = min(10, len(chunks))
        sampled_indices = random.sample(range(len(chunks)), num_samples)
        
        questions = []
        for sampled_index in sampled_indices:
            # Use the precomputed embedding for the sampled chunk
            query_embedding = embeddings[sampled_index].reshape(1, -1)
            # Search for top_k + 1 to exclude the query chunk itself
            top_k = 3
            D, I = data_index.search(query_embedding, top_k + 1)
            # Exclude the query chunk itself and take top_k
            relevant_indices = [idx for idx in I[0] if idx != sampled_index][:top_k]
            relevant_chunks = [chunks[idx] for idx in relevant_indices]
            # Generate questions
            chunk_questions = generate_questions(chunks[sampled_index]['chunk_text'], relevant_chunks)
            questions.extend(chunk_questions)
        
        return jsonify(questions), 200
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in /generate_full: {error_traceback}")
        return jsonify({'error': str(e), 'traceback': error_traceback}), 500



@app.route('/grade', methods=['post'])
def grade_questions():
    print("Processing questions")
    try:
        print("Processing questions")
        data = request.get_json()
        questions = data.get('questions')
        teacherId = data.get('teacherId')
        studentId = data.get('studentId')
        examId = data.get('examId')

        if not questions or not teacherId or not studentId or not examId:
            return jsonify({'error': 'Missing parameters'}), 400
        
        result = grade_pipeline(questions, teacherId, studentId, examId)

        return jsonify(result), 200
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in /grade: {error_traceback}")
        return jsonify({'error': str(e), 'traceback': error_traceback}), 500


if __name__ == '__main__':
    app.run(debug=True,port=8085,host="0.0.0.0")