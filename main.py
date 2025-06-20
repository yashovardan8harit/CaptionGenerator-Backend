from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import os
import time
import hashlib
from PIL import Image
import requests
from io import BytesIO
import random
from transformers import pipeline
from groq import Groq
import firebase_admin
from firebase_admin import credentials, auth
import sqlite3
from datetime import datetime
import json

# Load environment variables from a .env file
load_dotenv()

# --- CORRECTED FIREBASE ADMIN SDK INITIALIZATION ---
# This is the standard, recommended, and most reliable way.
# It uses the service account JSON file directly.

SERVICE_ACCOUNT_FILE = 'serviceAccountKey.json'

try:
    # Check if the file exists before trying to use it
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(f"'{SERVICE_ACCOUNT_FILE}' not found. Please download it from your Firebase project settings and place it in the backend directory.")

    # Initialize Firebase only if it hasn't been already
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred)
    print("✅ Firebase Admin SDK initialized successfully.")

except Exception as e:
    print(f"⚠️ Firebase Admin SDK initialization failed: {e}")
    print("Note: All features requiring user authentication will fail.")
# ----------------------------------------------------


# Initialize SQLite database for history
def init_database():
    conn = sqlite3.connect('caption_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS caption_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            image_url TEXT NOT NULL,
            basic_caption TEXT NOT NULL,
            enhanced_caption TEXT NOT NULL,
            style TEXT NOT NULL,
            custom_description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

# Initialize database on startup
init_database()

# Load BLIP captioning pipeline
print("🔄 Loading BLIP captioning pipeline...")
caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# CORS configuration for frontend dev environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update this if frontend URL changes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for caption generation request
class CaptionRequest(BaseModel):
    image_url: str
    style: str = "creative"
    custom_description: Optional[str] = None

class HistoryItem(BaseModel):
    id: int
    image_url: str
    basic_caption: str
    enhanced_caption: str
    style: str
    custom_description: Optional[str]
    created_at: str

# Dependency to verify Firebase token
async def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        token = authorization.split(" ")[1] if authorization.startswith("Bearer ") else authorization
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        print(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# --- All other functions (save_to_history, get_user_history, etc.) remain unchanged ---
# They are already well-written. I am leaving them as they were.

def save_to_history(user_id: str, image_url: str, basic_caption: str, enhanced_caption: str, style: str, custom_description: str = None):
    try:
        conn = sqlite3.connect('caption_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO caption_history (user_id, image_url, basic_caption, enhanced_caption, style, custom_description)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, image_url, basic_caption, enhanced_caption, style, custom_description))
        conn.commit()
        conn.close()
        print(f"✅ Saved caption to history for user: {user_id}")
    except Exception as e:
        print(f"❌ Error saving to history: {e}")

def get_user_history(user_id: str, limit: int = 50) -> List[dict]:
    try:
        conn = sqlite3.connect('caption_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, image_url, basic_caption, enhanced_caption, style, custom_description, created_at
            FROM caption_history 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
        rows = cursor.fetchall()
        conn.close()
        history = [dict(zip([column[0] for column in cursor.description], row)) for row in rows]
        return history
    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        return []

def enhance_caption_with_groq(basic_caption: str, style: str = "creative", custom_description: str = None) -> str:
    # This function is well-written and does not need changes.
    # ... (code for enhance_caption_with_groq as you provided) ...
    try:
        style_prompts = {
            "creative": "Transform this basic image description into a creative, engaging caption that tells a story or evokes emotion:",
            "funny": "Turn this image description into a humorous, witty caption that would make people smile:",
            "poetic": "Convert this image description into a beautiful, poetic caption with literary flair:",
            "marketing": "Transform this image description into compelling marketing copy that would grab attention:",
            "social": "Turn this into a perfect social media caption with personality and engagement:",
            "artistic": "Elevate this description into an artistic, sophisticated caption that appreciates the visual elements:"
        }
        
        # Handle custom style with user's specific description
        if style == "custom" and custom_description:
            prompt = f"""Create a caption for this image based on the user's specific request.\n\nBasic image description: "{basic_caption}"\n\nUser's specific request: "{custom_description}"\n\nGuidelines:\n- Follow the user's specific style/tone request as closely as possible\n- Keep it concise and engaging (1-3 sentences)\n- Make it suitable for social media (Instagram, Facebook, Twitter, YouTube thumbnails, etc.)\n- Add relevant emojis to enhance the caption\n- If the user's request is unclear, default to a creative engaging style\n- Don't just describe the image, create content that matches their request\n\nCaption:"""
        else:
            # Use predefined style prompts
            base_prompt = style_prompts.get(style, style_prompts["creative"])
            prompt = f"""{base_prompt}\n\nBasic description: "{basic_caption}"\n\nGuidelines:\n- Keep it concise (1-2 sentences)\n- Make it engaging and memorable\n- Don't just describe, add personality\n- Avoid overly dramatic language\n- Make it suitable for social media including instagram, facebook, twitter, youtube thumbnails, etc.\n- Also add emojis to the caption\n\nEnhanced caption:"""

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=200
        )
        
        enhanced_caption = chat_completion.choices[0].message.content.strip()
        
        # Clean up the response (remove any prefixes)
        prefixes_to_remove = ["Enhanced caption:", "Caption:", "Here's", "Here is"]
        for prefix in prefixes_to_remove:
            if enhanced_caption.lower().startswith(prefix.lower()):
                enhanced_caption = enhanced_caption[len(prefix):].lstrip(':').strip()
                break
        
        # Remove any remaining colons at the beginning
        if enhanced_caption.startswith(":"):
            enhanced_caption = enhanced_caption[1:].strip()
        
        return enhanced_caption
        
    except Exception as e:
        print(f"Error enhancing caption with Groq: {str(e)}")
        return basic_caption

# --- All of your endpoints are well-written and do not need changes ---
# I am including them here for completeness.

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {"message": "Enhanced Caption Generator API is running!"}

@app.get("/generate-signature")
def generate_signature():
    """
    Generates a Cloudinary signature using timestamp and secret,
    which is used for secure (signed) uploads.
    """
    timestamp = int(time.time())
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")

    if not (api_secret and api_key and cloud_name):
        return JSONResponse(
            status_code=500,
            content={"error": "Missing Cloudinary credentials in environment variables."}
        )

    # Parameters that will be sent with the upload (must match frontend)
    params_to_sign = {
        'folder': 'uploads',
        'timestamp': timestamp
    }
    
    # Sort parameters alphabetically by key
    sorted_params = sorted(params_to_sign.items())
    
    # Create parameter string: key1=value1&key2=value2
    params_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
    
    # Add API secret at the end
    string_to_sign = params_string + api_secret
    signature = hashlib.sha256(string_to_sign.encode()).hexdigest()

    return JSONResponse({
        "timestamp": timestamp,
        "signature": signature,
        "api_key": api_key,
        "cloud_name": cloud_name,
        "folder": "uploads"
    })

@app.post("/generate-caption")
def generate_caption(request: CaptionRequest, user_id: str = Depends(verify_token)):
    try:
        # ... (Your existing code for this endpoint) ...
        image_url = request.image_url
        style = request.style
        custom_description = request.custom_description

        if not image_url or not image_url.startswith("http"):
            raise HTTPException(status_code=400, detail="Invalid image URL")

        # Validate custom style has description
        if style == "custom" and (not custom_description or not custom_description.strip()):
            raise HTTPException(status_code=400, detail="Custom description is required when using custom style")

        # Fetch and process image
        response = requests.get(image_url)
        response.raise_for_status() # Raises an exception for bad status codes
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Generate basic caption with BLIP
        result = caption_pipeline(image)
        basic_caption = result[0]["generated_text"]

        # Enhance caption with Groq (including custom description if provided)
        enhanced_caption = enhance_caption_with_groq(basic_caption, style, custom_description)

        save_to_history(user_id, image_url, basic_caption, enhanced_caption, style, custom_description)

        response_data = {
            "success": True,
            "image_url": image_url,
            "basic_caption": basic_caption,
            "enhanced_caption": enhanced_caption,
            "style": style
        }

        # Include custom description in response for reference
        if style == "custom" and custom_description:
            response_data["custom_description"] = custom_description

        return JSONResponse(response_data)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate caption")

@app.get("/user/history")
def get_history(user_id: str = Depends(verify_token), limit: int = 50):
    try:
        history = get_user_history(user_id, limit)
        return JSONResponse({
            "success": True,
            "history": history,
            "total": len(history)
        })
    except Exception as e:
        print(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")

@app.delete("/user/history/{history_id}")
def delete_history_item(history_id: int, user_id: str = Depends(verify_token)):
    try:
        # ... (Your existing code for this endpoint) ...
        conn = sqlite3.connect('caption_history.db')
        cursor = conn.cursor()
        cursor.execute('SELECT user_id FROM caption_history WHERE id = ?', (history_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="History item not found")
        
        if result[0] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this item")
        
        cursor.execute('DELETE FROM caption_history WHERE id = ? AND user_id = ?', (history_id, user_id))
        conn.commit()
        conn.close()
        
        return JSONResponse({"success": True, "message": "History item deleted successfully"})
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting history item: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete history item")

@app.delete("/user/history")
def clear_all_history(user_id: str = Depends(verify_token)):
    try:
        # ... (Your existing code for this endpoint) ...
        conn = sqlite3.connect('caption_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM caption_history WHERE user_id = ?', (user_id,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return JSONResponse({"success": True, "message": f"Deleted {deleted_count} history items", "deleted_count": deleted_count})
    except Exception as e:
        print(f"Error clearing history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear history")

@app.get("/caption-styles")
def get_caption_styles():
    """Get available caption styles including the custom option"""
    return {
        "styles": [
            {"id": "creative", "name": "Creative", "description": "Engaging and imaginative"},
            {"id": "funny", "name": "Funny", "description": "Humorous and witty"},
            {"id": "poetic", "name": "Poetic", "description": "Beautiful and literary"},
            {"id": "marketing", "name": "Marketing", "description": "Compelling and attention-grabbing"},
            {"id": "social", "name": "Social Media", "description": "Perfect for social platforms"},
            {"id": "artistic", "name": "Artistic", "description": "Sophisticated and refined"},
            {"id": "custom", "name": "Custom", "description": "Describe your own style"}
        ]
    }

@app.get("/test-env")
def test_environment():
    """Test endpoint to check if environment variables are loaded"""
    return {
        "cloudinary_configured": bool(os.getenv("CLOUDINARY_API_KEY")),
        "cloud_name_exists": bool(os.getenv("CLOUDINARY_CLOUD_NAME")),
        "api_secret_exists": bool(os.getenv("CLOUDINARY_API_SECRET")),
        "groq_configured": bool(os.getenv("GROQ_API_KEY"))
    }

# Additional endpoint for testing custom descriptions
@app.post("/test-custom-caption")
def test_custom_caption(request: dict):
    """
    Test endpoint for custom caption generation
    Expects: {"basic_caption": "...", "custom_description": "..."}
    """
    try:
        basic_caption = request.get("basic_caption", "a person in a photo")
        custom_description = request.get("custom_description", "")
        
        if not custom_description:
            raise HTTPException(status_code=400, detail="Custom description is required")
        
        enhanced_caption = enhance_caption_with_groq(basic_caption, "custom", custom_description)
        
        return {
            "success": True,
            "basic_caption": basic_caption,
            "custom_description": custom_description,
            "enhanced_caption": enhanced_caption
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")