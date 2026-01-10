#!/usr/bin/env python3
"""
Test script to verify Gemini API connection and functionality
"""

import os
import json
import requests

# Set the API key
os.environ['GEMINI_API_KEY'] = 'AIzaSyCAKZxJCnt7BKfsBH1ImvunKuaui-2L_9U'

def test_gemini_api():
    """Test Gemini API with a simple fencing analysis request"""
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return False
    
    print(f"🔑 API Key configured: {api_key[:10]}...")
    
    # Simple test prompt
    test_prompt = """You are an expert AI fencing analyst named "Coach Sabre."

Your task is to analyze this test data and respond with a JSON object.

Test data: A fencer lost due to slow reaction time.

OUTPUT FORMAT:
```json
{
  "loss_category": "Test",
  "loss_sub_category": "Slow Reaction Test",
  "brief_reasoning": "This is a test response to verify API connectivity."
}
```"""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": test_prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "topK": 1,
            "topP": 0.8,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json"
        }
    }
    
    try:
        print("🚀 Testing Gemini API connection...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()

        candidates = result.get('candidates') or []
        if candidates:
            content_parts = []
            json_fragments = []
            for candidate in candidates:
                content_obj = candidate.get('content')
                if isinstance(content_obj, dict):
                    parts = content_obj.get('parts') or []
                elif isinstance(content_obj, list):
                    parts = content_obj
                else:
                    parts = []

                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if 'jsonValue' in part and part['jsonValue'] is not None:
                        json_fragments.append(part['jsonValue'])
                    if 'text' in part and isinstance(part['text'], str):
                        content_parts.append(part['text'])

                legacy_text = candidate.get('text') or candidate.get('output')
                if isinstance(legacy_text, str) and legacy_text.strip():
                    content_parts.append(legacy_text)

            if json_fragments:
                parsed_result = json_fragments[0] if len(json_fragments) == 1 else json_fragments
                print("✅ JSON parsing successful via jsonValue!")
                print(f"📊 Parsed result: {json.dumps(parsed_result, indent=2, ensure_ascii=False)}")
                return True

            if not content_parts:
                for candidate in candidates:
                    text_val = candidate.get('text') or candidate.get('output')
                    if isinstance(text_val, str) and text_val.strip():
                        content_parts.append(text_val)

            content = ''.join(content_parts).strip()
            print("✅ API Connection successful!")
            print(f"📝 Response: {content[:100]}...")
            
            # Try to parse as JSON
            try:
                parsed_result = json.loads(content)
                print("✅ JSON parsing successful!")
                print(f"📊 Parsed result: {json.dumps(parsed_result, indent=2, ensure_ascii=False)}")
                return True
                
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parsing failed: {e}")
                print(f"Raw content: {content}")
                return False
        else:
            print("❌ No candidates in API response")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Gemini API Test for Fencing Loss Analysis")
    print("=" * 50)
    
    success = test_gemini_api()
    
    if success:
        print("\n🎉 Gemini API is ready for fencing loss analysis!")
        print("✅ You can now use the video view loss analysis feature")
    else:
        print("\n❌ API test failed. Please check your API key and connection.")
