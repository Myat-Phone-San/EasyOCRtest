import easyocr
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import fitz # PyMuPDF
from PIL import Image
from io import BytesIO
import re 

# --- Configuration and Initialization ---

# Set the page configuration early (CORRECTED FUNCTION CALL)
st.set_page_config(
    page_title="Document Amendment OCR Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Specific Key Mappings for the Banking Form ---

# These are the exact text labels used for searching the document.
SEARCHABLE_KEYS = [
    "Applicant Name", "Documentary Credit No.", "Original Credit Amount", 
    "Contact Person / Tel", "Beneficiary Name",
    "Increse Credit Amount by", "Decrease Credit Amount by",
    "Other Amendments:", "New Latest Shipment date", "New Expiry Date",
    "Date", "CARGO", "Signature Verified By", "Input By", "Approved By", "Remarks",
    # Checkbox/Mode Fields (must be searchable separately)
    "Teletransmission", "Airmail", "Other mode (as stated below)", 
    "Our Account No", "Beneficiary"
]

# This is the final, fixed list of labels used for display (all fields must be present).
FINAL_KEY_LABELS = [
    "Applicant Name",
    "Documentary Credit No.",
    "Original Credit Amount",
    "Contact Person / Tel",
    "Beneficiary Name",
    "Amendment to be sent by (Teletransmission)",
    "Amendment to be sent by (Airmail)",
    "Amendment to be sent by (Other mode)",
    "Other mode details", 
    "Charges for (Our Account No)",
    "Account No / Value for Charges (Our Account No)", 
    "Charges for (Beneficiary)",
    "Increse Credit Amount by",
    "to (Final Increased Amount)", 
    "Decrease Credit Amount by",
    "to (Final Decreased Amount)", 
    "New Latest Shipment date",
    "New Expiry Date",
    "Other Amendments:",
    "Date (FOR BANK USE)", 
    "CARGO (FOR BANK USE)", 
    "Signature Verified By (FOR BANK USE)", 
    "Input By (FOR BANK USE)", 
    "Approved By (FOR BANK USE)", 
    "Remarks (FOR BANK USE)"
]

# --- Core OCR and Data Processing Functions ---

@st.cache_resource
def load_ocr_reader():
    """Initializes the EasyOCR reader once and caches it."""
    # Prioritize GPU, fall back to CPU
    try:
        reader = easyocr.Reader(['en'], gpu=True)
    except Exception:
        reader = easyocr.Reader(['en'], gpu=False, quantize=True)
    return reader

# Load the reader globally
reader = load_ocr_reader()

def get_center(bbox):
    """Helper function to find the center point of a bounding box."""
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    return (sum(x_coords) / 4, sum(y_coords) / 4)

def get_bbox_bottom_right(bbox):
    """Helper function to get the bottom-right coordinate."""
    return (bbox[2][0], bbox[2][1])

def group_and_sort_results(results):
    """Sorts OCR results by (Y, X) and groups them into logical rows."""
    sorted_results = sorted(results, key=lambda r: (get_center(r[0])[1], get_center(r[0])[0]))

    ROW_TOLERANCE = 20
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
                current_row.sort(key=lambda item: get_center(item[0])[0])
                grouped_rows.append(current_row)
                
                current_row = [r]
                baseline_y = center_y

        if current_row:
            current_row.sort(key=lambda item: get_center(item[0])[0])
            grouped_rows.append(current_row)
            
    return grouped_rows

def create_structured_dataframe(grouped_rows):
    """Converts grouped OCR results into a structured DataFrame, padding with '-'."""
    extracted_data = [[item[1] for item in row] for row in grouped_rows]
    
    max_cols = max([len(row) for row in extracted_data], default=0)
    padded_data = [row + ['-'] * (max_cols - len(row)) for row in extracted_data] 
    
    df = pd.DataFrame(padded_data, columns=[f'Col {i+1}' for i in range(max_cols)])
    return df

# --- Amount Parsing Helper ---
def parse_amount(amount_str):
    """Cleans common OCR noise and symbols for float conversion."""
    clean_str = re.sub(r'[^\d\.\,]', '', amount_str).replace(',', '').strip()
    try:
        return float(clean_str)
    except:
        return 0.0

# --- Core Extraction Logic (Refactored) ---

def extract_key_value_pairs(results):
    """
    Extracts Key-Value pairs based on proximity, mapping the results to the 
    fixed FINAL_KEY_LABELS list. Implements custom logic for the form's structure.
    """
    kv_data = {key: '-' for key in FINAL_KEY_LABELS}
    
    # Sort by Y-center, then X-center for organized processing
    sorted_results = sorted(results, key=lambda r: (get_center(r[0])[1], get_center(r[0])[0]))
    Y_PROXIMITY_TOLERANCE = 15
    X_MAX_DISTANCE = 400
    
    original_amount = 0.0
    original_currency = 'EUR'

    # --- Pass 1: Extract Main Fields (Horizontal/Vertical) ---
    for i, (bbox_key, text_key_raw, _) in enumerate(sorted_results):
        text_key = text_key_raw.strip()
        key_y_center = get_center(bbox_key)[1]
        key_x1 = bbox_key[0][0] # Key's left edge
        
        matched_search_key = next(
            (label for label in SEARCHABLE_KEYS if label in text_key), 
            None
        )
        
        if matched_search_key:
            
            # --- Key Mapping (for final output labels) ---
            final_key = matched_search_key
            if matched_search_key == "Teletransmission": final_key = "Amendment to be sent by (Teletransmission)"
            elif matched_search_key == "Airmail": final_key = "Amendment to be sent by (Airmail)"
            elif matched_search_key == "Other mode (as stated below)": final_key = "Amendment to be sent by (Other mode)"
            elif matched_search_key == "Our Account No": final_key = "Charges for (Our Account No)"
            elif matched_search_key == "Beneficiary": final_key = "Charges for (Beneficiary)"
            elif matched_search_key == "Date": final_key = "Date (FOR BANK USE)"
            elif matched_search_key == "CARGO": final_key = "CARGO (FOR BANK USE)"
            elif matched_search_key in ["Signature Verified By", "Input By", "Approved By", "Remarks"]:
                final_key = f"{matched_search_key} (FOR BANK USE)"
            
            # --- Checkbox Handling (Value is the checkmark itself) ---
            if final_key in [
                "Amendment to be sent by (Teletransmission)", "Amendment to be sent by (Airmail)",
                "Amendment to be sent by (Other mode)", "Charges for (Our Account No)", 
                "Charges for (Beneficiary)"
            ]:
                kv_data[final_key] = 'Yes/Checked'
                
                # Special logic to find the value next to the checkbox (e.g., account number or description)
                best_value = None
                min_distance = float('inf')
                
                for j, (bbox_val, text_val_raw, _) in enumerate(sorted_results):
                    if i == j: continue 
                    text_val = text_val_raw.strip()
                    val_y_center = get_center(bbox_val)[1]
                    val_x1 = bbox_val[0][0]
                    
                    is_on_same_line = abs(key_y_center - val_y_center) < Y_PROXIMITY_TOLERANCE
                    distance = val_x1 - get_bbox_bottom_right(bbox_key)[0] # Distance from key end to value start
                    
                    if is_on_same_line and distance > 5 and distance < 150:
                        if distance < min_distance:
                            min_distance = distance
                            best_value = text_val
                
                if final_key == "Amendment to be sent by (Other mode)" and best_value:
                    kv_data["Other mode details"] = best_value
                
                if final_key == "Charges for (Our Account No)" and best_value and "Our Account No" not in best_value:
                    # Filter out the key text itself which is sometimes read as the value
                    kv_data["Account No / Value for Charges (Our Account No)"] = best_value
                
                continue

            # --- Proximity Search for Value ---
            best_value = None
            min_distance = float('inf')
            
            for j, (bbox_val, text_val_raw, _) in enumerate(sorted_results):
                if i == j: continue 
                
                text_val = text_val_raw.strip()
                key_x2 = bbox_key[2][0] # Key's right edge
                val_x1 = bbox_val[0][0] # Value's left edge
                val_y_center = get_center(bbox_val)[1]
                
                is_on_same_line = abs(key_y_center - val_y_center) < Y_PROXIMITY_TOLERANCE
                
                # Check for value immediately below the key (e.g., Applicant Name, Contact Person)
                is_below = val_y_center > key_y_center and abs(val_y_center - key_y_center) < 40 and abs(val_x1 - key_x1) < 50
                
                # Horizontal value (Documentary Credit No, Original Credit Amount)
                is_to_right = val_x1 > key_x2
                distance = val_x1 - key_x2 if is_to_right else abs(val_y_center - key_y_center)

                if ((is_on_same_line and is_to_right and distance < X_MAX_DISTANCE) or is_below) and text_val not in text_key:
                    if distance < min_distance: 
                        min_distance = distance
                        best_value = text_val
                        
            if best_value and best_value != text_key: 
                kv_data[final_key] = best_value

            # Special handling for Original Credit Amount
            if final_key == "Original Credit Amount" and kv_data[final_key] != '-':
                # Expects a format like "EUR 85,000.00"
                parts = kv_data[final_key].split()
                if len(parts) >= 2 and parts[0].isalpha():
                    original_currency = parts[0]
                    original_amount = parse_amount(parts[1])
                else:
                    # Fallback for value without currency
                    original_amount = parse_amount(kv_data[final_key])

    # --- Pass 2: Complex / Calculated Fields & FOR BANK USE ---

    # 1. Increase/Decrease Amount and Calculation
    
    # Get the value from the OCR box for Increase
    increase_by_value_str = kv_data.get("Increse Credit Amount by", "-")
    # Clean and parse the value from the original image (e.g., "$50000 0 | $10000" might be read)
    # We assume the amount is the first valid number found
    match_inc = re.search(r'[\d\,\.]+', increase_by_value_str)
    increase_by_value = parse_amount(match_inc.group(0)) if match_inc else 0.0

    if increase_by_value > 0:
        final_inc_amount = original_amount + increase_by_value 
        kv_data["to (Final Increased Amount)"] = f"Result: {final_inc_amount:,.2f} {original_currency} (Currency Mix Warning)"
        kv_data["Increse Credit Amount by"] = f"${increase_by_value:,.2f}" 
    
    # Get the value from the OCR box for Decrease
    decrease_by_value_str = kv_data.get("Decrease Credit Amount by", "-")
    # Clean and parse the value from the original image (e.g., "EUR 80,00_o¢_EUR 80,000" might be read)
    match_dec = re.search(r'[\d\,\.]+', decrease_by_value_str)
    decrease_by_value = parse_amount(match_dec.group(0)) if match_dec else 0.0
    
    if decrease_by_value > 0:
        final_dec_amount = original_amount - decrease_by_value
        kv_data["to (Final Decreased Amount)"] = f"Result: {final_dec_amount:,.2f} {original_currency}"
        kv_data["Decrease Credit Amount by"] = f"{original_currency} {decrease_by_value:,.2f}"


    # 2. FOR BANK USE FIELDS (Specific box matching)
    
    # We iterate again to specifically link the key (like "Input By") to the value 
    # that is far below it in a fixed column structure.
    
    for i, (bbox_key, text_key_raw, _) in enumerate(sorted_results):
        text_key = text_key_raw.strip()
        key_y_center = get_center(bbox_key)[1]
        key_x_center = get_center(bbox_key)[0]
        
        # --- Approved By: General Manager & Deputy Manager ---
        if "Approved By" in text_key:
            approved_roles = []
            for j, (bbox_val, text_val_raw, _) in enumerate(sorted_results):
                text_val = text_val_raw.strip()
                val_y_center = get_center(bbox_val)[1]
                
                # Check for the roles located in the boxes below the 'Approved By' label
                if val_y_center > key_y_center and abs(val_y_center - key_y_center) < 100:
                    if "General Manager" in text_val or "Deputy Manager" in text_val:
                         approved_roles.append(text_val)
            
            if approved_roles:
                # Joining them as requested: General Manager, Deputy Manager
                kv_data["Approved By (FOR BANK USE)"] = ", ".join(approved_roles)
                
        # --- Signature Verified By (Handwritten Text) ---
        elif "Signature Verified By" in text_key:
            # Look for text immediately below the key
            for j, (bbox_val, text_val_raw, _) in enumerate(sorted_results):
                text_val = text_val_raw.strip()
                val_y_center = get_center(bbox_val)[1]
                
                # Look for text that is below the key and not just the key text repeated
                if val_y_center > key_y_center and abs(val_y_center - key_y_center) < 70 and text_val != text_key:
                    kv_data["Signature Verified By (FOR BANK USE)"] = text_val
                    break
        
        # --- Input By (Mg Myat Phone San) ---
        elif "Input By" in text_key:
            for j, (bbox_val, text_val_raw, _) in enumerate(sorted_results):
                text_val = text_val_raw.strip()
                val_y_center = get_center(bbox_val)[1]
                
                if val_y_center > key_y_center and abs(val_y_center - key_y_center) < 70 and ("Mg Myat Phone San" in text_val):
                    kv_data["Input By (FOR BANK USE)"] = text_val
                    break
                
        # --- Remarks (To grow with us) ---
        elif "Remarks" in text_key:
            for j, (bbox_val, text_val_raw, _) in enumerate(sorted_results):
                text_val = text_val_raw.strip()
                val_y_center = get_center(bbox_val)[1]
                
                if val_y_center > key_y_center and abs(val_y_center - key_y_center) < 70 and ("To grow with us" in text_val):
                    kv_data["Remarks (FOR BANK USE)"] = text_val
                    break
                
        # --- Other Amendments ---
        elif "Other Amendments:" in text_key:
            # Search for text in the large box below the key
            for j, (bbox_val, text_val_raw, _) in enumerate(sorted_results):
                text_val = text_val_raw.strip()
                val_y_center = get_center(bbox_val)[1]
                
                # Find text below the key within the general area of the box
                if val_y_center > key_y_center and abs(val_y_center - key_y_center) < 150 and text_val != text_key:
                    # This often captures the noise/value in that box
                    kv_data["Other Amendments:"] = text_val
                    break


    # 3. Final Assembly and cleanup
    
    # Ensure Original Credit Amount is correctly formatted
    if original_amount > 0:
        kv_data["Original Credit Amount"] = f"{original_currency} {original_amount:,.2f}"

    kv_df_list = []
    
    for label in FINAL_KEY_LABELS:
        kv_df_list.append({
            'Key Label (Form Text)': label,
            'Extracted Value': kv_data.get(label, '-') 
        })
        
    kv_df = pd.DataFrame(kv_df_list)
    return kv_df


def create_non_structured_text(grouped_rows):
    """Joins all detected text, structured by line, for the plain text output."""
    full_text = ""
    for row in grouped_rows:
        line = " ".join([item[1] for item in row])
        full_text += line + "\n"
    return full_text.strip()

def draw_boxes(image_array, results):
    """Draws bounding boxes and text on the image for visualization."""
    img_boxes = image_array.copy()
    # Convert BGR to RGB for EasyOCR and display
    img_rgb = cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB)
    
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        # Draw bounding box (Green)
        cv2.rectangle(img_rgb, tl, br, (0, 255, 0), 2)
        # Put text label (Blue)
        cv2.putText(img_rgb, text, (tl[0], tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Convert back to a PIL image for Streamlit display
    return Image.fromarray(img_rgb)

def process_ocr(image_array):
    """Performs OCR, sorts results, and structures the data."""
    if image_array is None:
        return [], [], None

    # EasyOCR expects RGB input
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Use EasyOCR to read text
    results = reader.readtext(img_rgb, detail=1)
    
    # Structure the results
    grouped_rows = group_and_sort_results(results)
    
    # Draw visualization
    img_with_boxes = draw_boxes(image_array, results)
    
    return results, grouped_rows, img_with_boxes

def handle_file_upload(uploaded_file):
    """Handles both image and PDF file uploads, converting them to an OpenCV array."""
    file_type = uploaded_file.type
    
    try:
        file_bytes = uploaded_file.read()

        if 'pdf' in file_type:
            with st.spinner("Converting PDF page 1 to image (150 DPI)..."):
                # Use PyMuPDF (fitz) to handle PDF conversion
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                if doc.page_count == 0:
                     st.error("Could not process PDF. The document is empty or unreadable.")
                     return None
                
                page = doc.load_page(0)
                DPI = 150
                zoom_factor = DPI / 72
                matrix = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                
                # Convert PyMuPDF pixmap to numpy array
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                # Ensure it's in BGR format for OpenCV compatibility if needed later
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) 
                doc.close()
                return img_array
            
        else: # Handle image files (jpg, png, etc.)
            img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img_array

    except Exception as e:
        st.error(f"Error loading file. Check if it's a valid Image or non-encrypted PDF. Error details: {e}")
        return None

# --- Download Utility ---

def get_download_button(data, is_dataframe, file_format, label, file_name_base, help_text=""):
    """Generates a common download button for different formats."""
    
    if is_dataframe:
        df = data
        if file_format == 'csv':
            data_out = df.to_csv(index=False).encode('utf-8')
            mime = 'text/csv'
            final_name = f'{file_name_base}.csv'
        else: # txt or doc
            # Use to_string for a plain text representation of the table
            data_out = df.to_string(index=False).encode('utf-8')
            mime = 'text/plain' if file_format == 'txt' else 'application/msword'
            final_name = f'{file_name_base}.{file_format}'
    else: # Raw Text
        data_out = data.encode('utf-8')
        mime = 'text/plain' if file_format == 'txt' else 'application/msword'
        final_name = f'{file_name_base}.{file_format}'
        
    st.download_button(
        label=label,
        data=data_out,
        file_name=final_name,
        mime=mime,
        help=help_text
    )

# --- Streamlit Application Layout ---

def main():
    st.title("📄 Document OCR Extractor")
    st.markdown("Upload the **Documentary Credit Amendment Application** form (image or PDF) to extract structured data.")
    
    # 1. File Upload
    uploaded_file = st.file_uploader(
        "Choose a Document File",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="For multi-page PDFs, only the first page will be processed."
    )

    st.markdown("---")

    image_array = None
    if uploaded_file is not None:
        st.info(f"File **'{uploaded_file.name}'** uploaded. Starting file conversion...")
        image_array = handle_file_upload(uploaded_file)

    # --- OCR Processing and Results Display ---

    if image_array is not None:
        st.subheader("2. OCR Processing and Result Formats")
        
        device_info = "GPU-Accelerated" if hasattr(reader, 'device') and reader.device != 'cpu' else "CPU-Optimized"
        spinner_message = f"🚀 Running {device_info} OCR and structuring data..."
            
        with st.spinner(spinner_message):
            results, grouped_rows, img_with_boxes = process_ocr(image_array)

            # Generate the required output formats
            df_structured = create_structured_dataframe(grouped_rows)
            df_kv_pairs = extract_key_value_pairs(results)
            non_structured_text = create_non_structured_text(grouped_rows) 

        if grouped_rows:
            # Display results in a two-column layout
            col_img, col_tabs = st.columns([1, 2])
            
            with col_img:
                st.markdown("### 🖼️ OCR Visualization (Bounding Boxes)")
                st.image(img_with_boxes, caption="Image with OCR Bounding Boxes", use_column_width=True)

            with col_tabs:
                # Match the required tab format
                tab1, tab2, tab3 = st.tabs(["🔑 KEY VALUE PAIRS", "📊 TABLES (Structured)", "📝 OCR (Line-by-line)"])
                
                # --- Tab 1: Key Value Pairs (The custom bank format) ---
                with tab1:
                    st.markdown("### Extracted Key-Value Pairs")
                    # Displaying the custom, refined output
                    st.dataframe(df_kv_pairs[['Key Label (Form Text)', 'Extracted Value']], use_container_width=True, hide_index=True)
                    
                    st.markdown("#### Download Options")
                    col_csv, col_txt, col_word = st.columns(3)
                    
                    with col_csv:
                        get_download_button(df_kv_pairs[['Key Label (Form Text)', 'Extracted Value']], True, 'csv', "📥 Download CSV", 'key_value_pairs')

                    with col_txt:
                        get_download_button(df_kv_pairs[['Key Label (Form Text)', 'Extracted Value']], True, 'txt', "📥 Download TXT", 'key_value_pairs')
                        
                    with col_word:
                        get_download_button(df_kv_pairs[['Key Label (Form Text)', 'Extracted Value']], True, 'doc', "📥 Download DOC (Word)", 'key_value_pairs', help_text="Saves the table data as a text file with a .doc extension.")
                
                # --- Tab 2: Structured Table (General OCR output) ---
                with tab2:
                    st.markdown("### General OCR Output (Grouped by Row, Padded with '-')")
                    st.dataframe(df_structured, use_container_width=True)
                    
                    st.markdown("#### Download Options")
                    col_csv, col_txt, col_word = st.columns(3)
                    
                    with col_csv:
                        get_download_button(df_structured, True, 'csv', "📥 Download CSV", 'structured_ocr_data')

                    with col_txt:
                        get_download_button(df_structured, True, 'txt', "📥 Download TXT", 'structured_ocr_data')
                        
                    with col_word:
                        get_download_button(df_structured, True, 'doc', "📥 Download DOC (Word)", 'structured_ocr_data')
                
                # --- Tab 3: All Non-Structured Text (Raw text) ---
                with tab3:
                    st.markdown("### Full Extracted Text (Sorted by Reading Order)")
                    st.text_area("Non-Structured Text", non_structured_text, height=400)
                    
                    st.markdown("#### Download Options")
                    col_txt, col_word, _ = st.columns(3) 
                    
                    with col_txt:
                        get_download_button(non_structured_text, False, 'txt', "📥 Download TXT", 'full_raw_text')
                        
                    with col_word:
                        get_download_button(non_structured_text, False, 'doc', "📥 Download DOC (Word)", 'full_raw_text')
                        
        else:
            st.warning("No text could be extracted from the document. Please ensure the image is clear or the PDF is non-empty.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Built with EasyOCR, OpenCV, Pandas, Streamlit, and PyMuPDF.
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()