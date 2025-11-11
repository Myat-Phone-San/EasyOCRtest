import easyocr
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from io import BytesIO

# --- Configuration and Initialization ---

# Set the page configuration early
st.set_page_config(
    page_title="EasyOCR File Scanner",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Optimized Core Functions ---

@st.cache_resource
def load_ocr_reader():
    """
    Initializes the EasyOCR reader once and caches it. 
    Attempts GPU mode first, and falls back to CPU-optimized mode if GPU fails.
    """
    
    
    
    try:
        # 1. Primary Attempt: GPU Acceleration (gpu=True)
        reader = easyocr.Reader(['en'], gpu=True) 
        
    except Exception as e:
        # This block executes if GPU initialization fails (e.g., no CUDA setup)
        st.warning(f"⚠️ GPU initialization failed. Falling back to CPU-optimized mode. Error details: {e}")
        
        try:
            # 2. Fallback Attempt: CPU Optimized (gpu=False, quantize=True)
            reader = easyocr.Reader(['en'], gpu=False, quantize=True)
           
        except Exception as e_f:
            # 3. Last Resort: Default CPU mode (if optimized fails)
            st.error(f"❌ CPU-optimized model initialization failed. Using default CPU mode. Error details: {e_f}")
            reader = easyocr.Reader(['en'], gpu=False)

    return reader

# Load the OCR reader
reader = load_ocr_reader()

def get_center(bbox):
    """Helper function to find the center point of a bounding box."""
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    return (sum(x_coords) / 4, sum(y_coords) / 4)

def process_ocr(image_array):
    """
    Performs OCR, sorts results, and structures the data into a DataFrame.
    Returns (DataFrame, image_with_boxes)
    """
    if image_array is None:
        return pd.DataFrame([["Error", "No valid image provided."]]), None

    # Convert to RGB for EasyOCR (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Run OCR
    # box: coordinates, text: detected text, prob: confidence score
    results = reader.readtext(img_rgb, detail=1) 
    
    # Sort the results primarily by Y-coordinate, then by X-coordinate
    sorted_results = sorted(results, key=lambda r: (get_center(r[0])[1], get_center(r[0])[0]))

    # Group into logical rows
    ROW_TOLERANCE = 20 # Max vertical distance allowed to consider text on the same line
    grouped_rows = []
    current_row = []

    if sorted_results:
        current_row.append(sorted_results[0])
        baseline_y = get_center(sorted_results[0][0])[1]

        for i in range(1, len(sorted_results)):
            r = sorted_results[i]
            center_y = get_center(r[0])[1]

            if abs(center_y - baseline_y) < ROW_TOLERANCE:
                current_row.append(r)
            else:
                # Sort the completed row by X position before appending
                current_row.sort(key=lambda item: get_center(item[0])[0])
                grouped_rows.append(current_row)
                
                # Start a new row
                current_row = [r]
                baseline_y = center_y

        # Handle the last row
        if current_row:
            current_row.sort(key=lambda item: get_center(item[0])[0])
            grouped_rows.append(current_row)
    
    # Convert grouped rows to DataFrame
    extracted_data = [[item[1] for item in row] for row in grouped_rows]
    
    # Use the longest row to define the columns for better alignment visualization
    max_cols = max([len(row) for row in extracted_data], default=0)
    df = pd.DataFrame(extracted_data, columns=[f'Column {i+1}' for i in range(max_cols)])

    # Create visualization image with bounding boxes
    img_boxes = image_array.copy()
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        # Draw green bounding box
        cv2.rectangle(img_boxes, tl, br, (0, 255, 0), 2) 
        # Put blue text (slightly above box)
        cv2.putText(img_boxes, text, (tl[0], tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Convert OpenCV BGR to PIL Image (RGB) for Streamlit display
    img_pil = Image.fromarray(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
    
    return df, img_pil

def handle_file_upload(uploaded_file):
    """Handles both image and PDF file uploads, converting them to an OpenCV array."""
    file_type = uploaded_file.type
    
    try:
        file_bytes = uploaded_file.read()

        if 'pdf' in file_type:
            with st.spinner("Converting PDF page 1 to image... (DPI 150)"):
                pdf_buffer = BytesIO(file_bytes)
                pages = convert_from_bytes(pdf_buffer.read(), first_page=1, last_page=1, dpi=150)
                
                if pages:
                    source_image = pages[0]
                    img_array = np.array(source_image.convert('RGB')) 
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) 
                    return img_array
                else:
                    st.error("Could not process PDF page. Ensure the PDF is not encrypted.")
                    return None
            
        else: # Handle image files
            img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img_array

    except Exception as e:
        st.error(f"Error loading file. Check if it's a valid Image or non-encrypted PDF. Error details: {e}")
        return None

# --- Streamlit Application Layout ---

st.title("📄 EasyOCR Document Scanner (File Upload)")
st.markdown("Upload a document image (.jpg, .png) or a single-page PDF to extract and structure the text.")

image_array = None
uploaded_file = st.file_uploader(
    "Choose a Document File", 
    type=['jpg', 'jpeg', 'png', 'pdf'],
    help="For multi-page PDFs, only the first page will be processed."
)

if uploaded_file is not None:
    st.info(f"File **'{uploaded_file.name}'** uploaded. Starting file conversion...")
    image_array = handle_file_upload(uploaded_file)

st.markdown("---")

# --- OCR Processing and Results Display ---

if image_array is not None:
    st.subheader("2. OCR Processing Results")
    
    # Use a dynamic message based on the initialization result
    if hasattr(reader, 'device') and reader.device == 'cpu':
        spinner_message = "🧠 Running CPU-Optimized OCR and structuring data..."
    else:
        spinner_message = "🚀 Running GPU-Accelerated OCR and structuring data..."
        
    with st.spinner(spinner_message):
        df_result, img_with_boxes = process_ocr(image_array)

    if not df_result.empty and img_with_boxes is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🖼️ OCR Visualization")
            st.image(img_with_boxes, caption="Image with OCR Bounding Boxes", use_column_width=True)
            
        with col2:
            st.markdown("### 📊 Extracted Structured Data")
            # Display the DataFrame
            st.dataframe(df_result, use_container_width=True)
            
            # Create a CSV for download
            csv_data = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Extracted Data as CSV",
                data=csv_data,
                file_name='extracted_data.csv',
                mime='text/csv',
                help="Download the text structured into rows and columns."
            )
    else:
        st.warning("No text could be extracted from the document. Please ensure the image is clear or the PDF is non-empty.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with EasyOCR, OpenCV, Pandas, and Streamlit.
</div>
""", unsafe_allow_html=True)