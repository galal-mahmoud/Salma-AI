import cv2
import numpy as np
from PIL import Image
import os
import base64
import google.generativeai as genai
import json
import dotenv
import time
dotenv.load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("API_KEY is missing. Please set it in the .env file.")

genai.configure(api_key=API_KEY)
def detect_circles(image_path,teacherId,examId,studentId,questionId,get_rectangles=False):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    if not os.path.exists("imgs"):
        os.makedirs("imgs")

    if not teacherId or not examId or not studentId or not questionId:
        raise ValueError("TeacherId, ExamId and StudentId are required")
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Define the region of interest (left side of image)
    height, width = binary.shape
    roi_width = width // 8  # Adjust this value based on where circles are
    roi = binary[:, :roi_width]
    
    # Find contours
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for contour in contours:
        # Filter contours based on area and circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Adjust these thresholds based on your image
        if area > 15 and circularity > 0.6:  # Minimum area and circularity threshold
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circles.append((int(x), int(y), int(radius)))
    

    roi_width = width // 8  # Adjust this value based on where circles are
    # Define the region of interest (right side of image)
    roi = binary[:, -roi_width:]
    
    # Find contours    
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter contours based on area and circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Adjust these thresholds based on your image
        if area > 15 and circularity > 0.6:  # Minimum area and circularity threshold
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circles.append((width - roi_width + int(x), int(y), int(radius)))
    # Sort circles by y-coordinate
    circles.sort(key=lambda x: x[1])
    if get_rectangles:
        os.makedirs(os.path.join("imgs",f"{teacherId}"),exist_ok=True)
        os.makedirs(os.path.join("imgs",f"{teacherId}",f"{examId}"),exist_ok=True)
        os.makedirs(os.path.join("imgs",f"{teacherId}",f"{examId}",f"{studentId}"),exist_ok=True)
        os.makedirs(os.path.join("imgs",f"{teacherId}",f"{examId}",f"{studentId}",f"{questionId}"),exist_ok=True)

        # Draw rectangle from the x of the first circle to the x of the next circle and min y and max y
        for i in range(0, len(circles)-1, 2):
            x1 = min(circles[i][0],circles[i+1][0])
            x2 = max(circles[i][0],circles[i+1][0])
            y1 = min(circles[i][1], circles[i+1][1]) - 30
            y2 = max(circles[i][1], circles[i+1][1]) 
            # cv2.rectangle(image, (x1, y1-5), (x2, y2+5), (0, 255, 0), 2)
            rec = image[y1-15:y2+15,x1:x2]
            #threshhold the rec
            rec = cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY)
            _, rec = cv2.threshold(rec, 245, 255, cv2.THRESH_BINARY_INV)
            # bitwise not the rec
            
            
            # Detect lines using Hough Transform
            # lines = cv2.HoughLinesP(rec, 1, np.pi / 180, threshold=4, minLineLength=width //2, maxLineGap=10)
            
            # if lines is not None:
            #     print(f"Found {len(lines)} lines")
            #     for line in lines:
            #         x1, y1, x2, y2 = line[0]
            #         cv2.line(rec, (x1, y1), (x2, y2), (0, 255, 0), 2)
            rec = cv2.bitwise_not(rec)
            cv2.imwrite(os.path.join("imgs",f"{teacherId}",f"{examId}",f"{studentId}",f"{questionId}",f"line_{i // 2 + 1}.jpg"),rec)
            result= None

    else:
        if len(circles) >= 2:
            y_diffs = [abs(circles[i+1][1] - circles[i][1]) for i in range(0, len(circles)-1, 2)]
            average_tilt = sum(y_diffs) / len(y_diffs)
            print(f"Average tilt: {average_tilt}")
        else:
            print("Not enough circles to calculate tilt.")

        
        
        # Draw circles on image copy
        result = image.copy()
        # for (x, y, r) in circles:
        #     cv2.circle(result, (int(x), int(y)), int(r), (0, 255, 0), 2)
        
        # Rotate the image to fix the tilt
        (h, w) = result.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -0.02 * average_tilt, 1.0)
        rotated_image = cv2.warpAffine(result, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        result = rotated_image

    return result, len(circles)

def process_image(image_path,teacherId,examId,studentId,questionId,get_rectangles=False):
    
    if not teacherId or not examId or not studentId:
        raise ValueError("TeacherId, ExamId and StudentId are required")
    
    result_image, circle_count = detect_circles(image_path,teacherId,examId,studentId,questionId, get_rectangles)
    # Save the result
    if result_image is None:
        return
    
    output_path = image_path
    cv2.imwrite(output_path, result_image)
    
    print(f"Found {circle_count} circles")
    print(f"Result saved as {output_path}")

def extract_txt(image_path):

    if not os.path.exists(image_path):
        raise ValueError(f"Folder not found at {image_path}")
    
   
    
    # # Load the model and processor
    # processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
    # model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')


    # # Load three images at once and convert to RGB
    image  =Image.open(image_path).convert("RGB") 

    # System prompt for expert OCR in Arabic
    system_prompt = """
    You are an expert OCR (Optical Character Recognition) model. Your job is to extract and return only the text (Arabic or English) from the provided images.
    Do not provide any explanation or formatting, just output the recognized text as plain text.
    """
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel('gemini-2.5-flash')

    # Send all images with the system prompt to the model
    time.sleep(5)
    result = model.generate_content(
        contents=[system_prompt, image,'do ocr to this image'],
        generation_config=genai.GenerationConfig(
            temperature=0.0,
        )
    )
    print(result)
    return result.candidates[0].content.parts[0].text


def check_spelling(user_text):
    # Define the prompt
    prompt = """
    You are a helpful assistant that fixes spelling mistakes, missing words, and incorrect words in text generated by an OCR model. 
    The text may contain errors due to OCR inaccuracies. Your task is to correct the text without changing its meaning or rephrasing it.
    
    Instructions:
    1. Fix spelling mistakes.
    2. Add missing words if necessary.
    3. Correct wrong words.
    4. Do not change the meaning or rephrase the text.
    5. Output only the corrected text in a JSON format with the key "corrected_text".
    
    OCR Text: {user_text}
    """

    # Combine the prompt with the user input
    full_prompt = prompt.format(user_text=user_text)

    # Initialize the model
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Generate the corrected text
    time.sleep(5)
    result = model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            response_mime_type="application/json",
        )
    )

    # Return the corrected text from the JSON response
    return result.candidates[0].content.parts[0].text


def grade(student_answer, ideal_answer):
    # Define the prompt
    student_answer = json.loads(student_answer).get("corrected_text")
    prompt = """
    You are a helpful assistant that grades student answers based on an ideal answer. 
    Your task is to evaluate the student's answer strictly against the ideal answer and provide a grade out of 10 along with an explanation.

    Instructions:
    1. Grade the student's answer based on how well it matches the ideal answer.
    2. Do not consider any external information or assumptions.
    3. Provide a grade as a float out of 10.
    4. Provide a clear explanation of why the student received that grade.
    5. Output the result in JSON format with the keys "grade" and "explanation".
    6. The explanation should be concise and directly related to the student's answer and mention it if it's short.
    Ideal Answer: {ideal_answer}

    Student Answer: {student_answer}
    """

    # Combine the prompt with the inputs
    full_prompt = prompt.format(
        ideal_answer=ideal_answer,
        student_answer=student_answer
    )

    # Initialize the model
    model = genai.GenerativeModel('gemini-2.5-flash')
    # Generate the grading result
    time.sleep(5)
    result = model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,  # Set temperature to 0 for deterministic grading
            response_mime_type="application/json",
        )
    )

    # Parse the JSON response
    response = result.candidates[0].content.parts[0].text
    return response

def grade_pipeline(questions,teacherId,examId,studentId):
    # Grade the student's answer
    if not questions:
        raise ValueError("No images found in the input data")
    
    if not teacherId or not examId or not studentId:
        raise ValueError("TeacherId, ExamId and StudentId are required")

    # Decode base64 images and store them
    
    os.makedirs("imgs", exist_ok=True)
    os.makedirs(os.path.join("imgs", teacherId), exist_ok=True)
    os.makedirs(os.path.join("imgs", teacherId, examId), exist_ok=True)
    os.makedirs(os.path.join("imgs", teacherId, examId, studentId), exist_ok=True)
    student_path = os.path.join("imgs", teacherId, examId, studentId)

    decoded_images = []
    answers = {}
    for i, question in enumerate(questions):
        img, questionId, questionAnswer = question["img"], question["questionId"], question["questionAnswer"]
        answers[questionId] = questionAnswer
        print("CAME HERE 1")
        img_data = base64.b64decode(img)
        img_array = np.frombuffer(img_data, np.uint8)
        img_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        print("CAME HERE 2")
        decoded_images.append(img_data)
        os.makedirs(os.path.join(student_path,questionId), exist_ok=True)    
        os.makedirs(os.path.join(student_path,questionId,"full"), exist_ok=True)    
        cv2.imwrite(os.path.join(student_path,questionId, "full", f"image_{i}.jpg"), img_data)
            
    
        

    # Process each image and extract text
    grades = []
    for q_id in os.listdir(student_path):
        full_q_path = os.path.join(student_path, q_id, "full")
        for img_path in os.listdir(full_q_path):
            extracted_text = extract_txt(os.path.join(full_q_path, img_path))
            corrected_text = check_spelling(extracted_text)
            print(f"Extracted text for {q_id}: {extracted_text}")
            print(f"Corrected text for {q_id}: {corrected_text}")
            response = grade(corrected_text,answers[q_id])
            print( "corrected_text: ",corrected_text)
            print("answer: ",answers[q_id])
            if isinstance(corrected_text, str):
                corrected_text = json.loads(corrected_text).get("corrected_text", corrected_text)
            elif isinstance(corrected_text, dict):
                corrected_text = corrected_text.get("corrected_text", corrected_text)
                
            grades.append((q_id, json.loads(response).get("grade"), json.loads(response).get("explanation"), corrected_text, answers[q_id]))
            print(f"Graded {q_id}: {json.loads(response).get('grade')} - {json.loads(response).get('explanation')}, corrected_text: {corrected_text}, ideal_answer: {answers[q_id]}")
    

    # Return the grade and explanation
    return grades