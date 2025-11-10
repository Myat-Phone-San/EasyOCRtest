import easyocr
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import fitz # PyMuPDF
from PIL import Image
from io import BytesIO

# --- Configuration and Initialization ---

# Set the page configuration early
st.set_page_config(
    page_title="EasyOCR Structured & Non-Structured Scanner",
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

def group_and_sort_results(results):
    """
    Sorts OCR results by (Y, X) and groups them into logical rows.
    Returns: list of rows, where each row is a list of (bbox, text, prob) tuples.
    """
    
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
            
    return grouped_rows

def create_structured_dataframe(grouped_rows):
    """Converts grouped OCR results into a structured DataFrame."""
    extracted_data = [[item[1] for item in row] for row in grouped_rows]
    
    # Use the longest row to define the columns for better alignment visualization
    max_cols = max([len(row) for row in extracted_data], default=0)
    df = pd.DataFrame(extracted_data, columns=[f'Column {i+1}' for i in range(max_cols)])
    return df

def create_non_structured_text(grouped_rows):
    """
    Joins all detected text, structured by line, for the plain text output (overall texts).
    """
    full_text = ""
    for row in grouped_rows:
        line = " ".join([item[1] for item in row])
        full_text += line + "\n"
    return full_text.strip()


def draw_boxes(image_array, results):
    """Draws bounding boxes and text on the image for visualization."""
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
    return Image.fromarray(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))


def process_ocr(image_array):
    """
    Performs OCR, sorts results, and structures the data.
    Returns: (full_results, grouped_rows, image_with_boxes)
    """
    if image_array is None:
        return [], [], None

    # Convert to RGB for EasyOCR
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Run OCR
    results = reader.readtext(img_rgb, detail=1)
    
    # Group and sort results
    grouped_rows = group_and_sort_results(results)

    # Create visualization image
    img_with_boxes = draw_boxes(image_array, results)
    
    return results, grouped_rows, img_with_boxes


def handle_file_upload(uploaded_file):
    """
    Handles both image and PDF file uploads, converting them to an OpenCV array.
    Uses PyMuPDF (fitz) for PDF handling.
    """
    file_type = uploaded_file.type
    
    try:
        file_bytes = uploaded_file.read()

        if 'pdf' in file_type:
            with st.spinner("Converting PDF page 1 to image using PyMuPDF (150 DPI)..."):
                
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                if doc.page_count == 0:
                     st.error("Could not process PDF. The document is empty or unreadable.")
                     return None
                
                page = doc.load_page(0)
                
                DPI = 150
                zoom_factor = DPI / 72
                matrix = fitz.Matrix(zoom_factor, zoom_factor)
                
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                
                # Convert RGB (from pixmap) to BGR (for OpenCV)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                doc.close()
                return img_array
            
        else: # Handle image files
            img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img_array

    except Exception as e:
        st.error(f"Error loading file. Check if it's a valid Image or non-encrypted PDF. Error details: {e}")
        return None

# --- Streamlit Application Layout ---

st.title("📄 EasyOCR Document Scanner")
st.markdown("Upload a document image (.jpg, .png) or a single-page PDF to extract text and structure it.")

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
    st.subheader("2. OCR Processing and Result Formats")
    
    # Use a dynamic message based on the initialization result
    device_info = "GPU-Accelerated" if hasattr(reader, 'device') and reader.device != 'cpu' else "CPU-Optimized"
    spinner_message = f"🚀 Running {device_info} OCR and structuring data..."
        
    with st.spinner(spinner_message):
        # The main OCR run
        results, grouped_rows, img_with_boxes = process_ocr(image_array)

        # Generate the two required output formats
        df_structured = create_structured_dataframe(grouped_rows)
        non_structured_text = create_non_structured_text(grouped_rows) 

    if grouped_rows:
        col_img, col_tabs = st.columns([1, 2])
        
        with col_img:
            st.markdown("### 🖼️ OCR Visualization")
            st.image(img_with_boxes, caption="Image with OCR Bounding Boxes", use_column_width=True)

        with col_tabs:
            # Only two tabs now: Structured Table and Non-Structured Text
            tab1, tab2 = st.tabs(["📊 Structured Table", "📝 All Non-Structured Text"])
            
            # --- Tab 1: Structured Table ---
            with tab1:
                st.markdown("### Text Grouped by Row and Column")
                st.dataframe(df_structured, use_container_width=True)
                
                st.markdown("#### Download Options")
                col_csv, col_txt, col_word = st.columns(3)
                
                # --- Structured Download Options ---
                with col_csv:
                    # CSV Download
                    csv_data = df_structured.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name='structured_data.csv',
                        mime='text/csv'
                    )

                with col_txt:
                    # Plain Text Download 
                    text_from_df = df_structured.to_string(index=False)
                    st.download_button(
                        label="📥 Download TXT",
                        data=text_from_df.encode('utf-8'),
                        file_name='structured_data.txt',
                        mime='text/plain'
                    )
                    
                with col_word:
                    # Word Download (saved as .doc)
                    st.download_button(
                        label="📥 Download DOC (Word)",
                        data=text_from_df.encode('utf-8'),
                        file_name='structured_data.doc',
                        mime='application/msword',
                        help="Saves the table data as a text file with a .doc extension."
                    )
            
            # --- Tab 2: All Non-Structured Text ---
            with tab2:
                st.markdown("### Full Extracted Text (Sorted by Read Order)")
                st.text_area("Non-Structured Text", non_structured_text, height=400)
                
                st.markdown("#### Download Options")
                col_txt, col_word, _ = st.columns(3) 
                
                # --- Non-Structured Download Options ---
                with col_txt:
                    # Plain Text Download
                    st.download_button(
                        label="📥 Download TXT",
                        data=non_structured_text.encode('utf-8'),
                        file_name='full_text.txt',
                        mime='text/plain'
                    )
                with col_word:
                    # Word-compatible Download
                    st.download_button(
                        label="📥 Download DOC (Word)",
                        data=non_structured_text.encode('utf-8'),
                        file_name='full_text.doc',
                        mime='application/msword'
                    )
    else:
        st.warning("No text could be extracted from the document. Please ensure the image is clear or the PDF is non-empty.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with EasyOCR, OpenCV, Pandas, Streamlit, and **PyMuPDF**.
</div>
""", unsafe_allow_html=True)
