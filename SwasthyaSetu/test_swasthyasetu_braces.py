#!/usr/bin/env python3
"""
Test script for SwasthyaSetu T5 endpoint with braces data
"""

import json
import time
import requests

# Your exact braces JSON data
braces_data = [
    {
        "uri": "3392171",
        "question": "do braces hurt????",
        "context": "pain?\nhard to talk?",
        "answers": [
            "yes yes yes. But not horribly painful. you get used to them and they become easier to talk with. the pain is only when they get tightened and until you get used to them. They give you wax to put around the spots that hurt you when your tongue rubs against the brackets and you won't feel the pain.",
            "They hurt for a bit when you first get them.They feel tight.Then, they settle down. However, they hurt each time you get them adjusted for a few days afterward. But,dont worry,you'll get used to the pain.",
            "yes yes",
            "they hurt when u first get them but then u get used 2 them really  easily. sometimes u like drool or spit or something but unless u have a pallete expander u can talk normal. if u have 1 ur \"K's\" get all screwed up.",
            "Yes THEy HuRt WhEn YoU fIrSt GeT them but just for a few days...If you eat something hard and it gets stuck in them it mite hurt...and also after youy get them tightened..but they do not make it hard to talk, only retainers do that!!!",
            "at the begining you feel really weird and it hurts a little bit but then you get used to them\nis not hard to talk at all you can talk normally",
            "I had them...they hurt for a bit when you first get them...feel tight.  then, they settle down.  However, they hurt each time you get them adjusted for a few days afterward.",
            "I just got my braces a few days ago. They felt like they were loose for the first two days, and it was really hard to eat. I got over speaking difficulties quickly, although the inside of your mouth gets kind of scratched and torn- don't fret, it heals quickly enough. After that you get calouses, so it doesn't hurt anymore. I think the pain felt depends on how old you are when you get them, what kind of treatment you are getting, and how severe your bite is. Lately, I've forgotten I'd had them on. You get used to them.",
            "Intially they are painful...also depending on your age and the condition your teeth are in. I have had my for almost 2 1/2 years and my first year was painful. Now I forget I have them...they are coming off soon and I can't wait! I recommend Aleve, its really helpful in relieving the pain."
        ],
        "labelled_answer_spans": {
            "EXPERIENCE": [
                {"txt": ": yes yes yes. But not horribly painful. you get used to them and they become easier to talk with. the pain is only when they get tightened and until you get used to them. They give you wax to put around the spots that hurt you when your tongue rubs against the brackets and you won't feel the pain", "label_spans": [79, 377]},
                {"txt": "They hurt for a bit when you first get them.They feel tight.Then, they settle down. However, they hurt each time you get them adjusted for a few days afterward. But,dont worry,you'll get used to the pain.", "label_spans": [389, 593]},
                {"txt": "they hurt when u first get them but then u get used 2 them really  easily. sometimes u like drool or spit or something but unless u have a pallete expander u can talk normal. if u have 1 ur \"K's\" get all screwed up", "label_spans": [622, 836]},
                {"txt": "Yes THEy HuRt WhEn YoU fIrSt GeT them but just for a few days...If you eat something hard and it gets stuck in them it mite hurt...and also after youy get them tightened..but they do not make it hard to talk, only retainers do that!", "label_spans": [848, 1080]},
                {"txt": "t the begining you feel really weird and it hurts a little bit but then you get used to themis not hard to talk at all you can talk normally", "label_spans": [1094, 1234]},
                {"txt": "had them...they hurt for a bit when you first get them...feel tight.  then, they settle down.  However, they hurt each time you get them adjusted for a few days afterward.", "label_spans": [1247, 1418]},
                {"txt": "I just got my braces a few days ago. They felt like they were loose for the first two days, and it was really hard to eat. I got over speaking difficulties quickly, although the inside of your mouth gets kind of scratched and torn- don't fret, it heals quickly enough. After that you get calouses, so it doesn't hurt anymore. I think the pain felt depends on how old you are when you get them, what kind of treatment you are getting, and how severe your bite is. Lately, I've forgotten I'd had them on. You get used to them.", "label_spans": [1429, 1953]},
                {"txt": "Intially they are painful...also depending on your age and the condition your teeth are in. I have had my for almost 2 1/2 years and my first year was painful. Now I forget I have them...they are coming off soon and I can't wait! I recommend Aleve, its really helpful in relieving the pain.", "label_spans": [1964, 2254]}
            ]
        },
        "labelled_summaries": {
            "EXPERIENCE_SUMMARY": "The various experiences with braces are shared, with some individuals describing initial pain and discomfort, especially during adjustments or the first few days after getting them. However, a common theme emerges that people tend to get used to the discomfort over time. The pain is often associated with tightness, difficulty eating, and potential challenges in speaking initially. Some mention the use of wax to alleviate pain caused by friction with brackets. Despite the initial challenges, many report adapting quickly and experiencing less discomfort as time goes on. The experiences also highlight that age, the specific treatment plan, and the severity of the dental issues can influence the level of discomfort. Ultimately, several individuals express that the pain diminishes over time, and they become accustomed to having braces."
        },
        "raw_text": "uri: 3392171\nquestion: do braces hurt????\ncontext: pain?\nhard to talk?\nanswer_0: yes yes yes. But not horribly painful. you get used to them and they become easier to talk with. the pain is only when they get tightened and until you get used to them. They give you wax to put around the spots that hurt you when your tongue rubs against the brackets and you won't feel the pain.\nanswer_1: They hurt for a bit when you first get them.They feel tight.Then, they settle down. However, they hurt each time you get them adjusted for a few days afterward. But,dont worry,you'll get used to the pain.\nanswer_2: yes yes\nanswer_3: they hurt when u first get them but then u get used 2 them really  easily. sometimes u like drool or spit or something but unless u have a pallete expander u can talk normal. if u have 1 ur \"K's\" get all screwed up.\nanswer_4: Yes THEy HuRt WhEn YoU fIrSt GeT them but just for a few days...If you eat something hard and it gets stuck in them it mite hurt...and also after youy get them tightened..but they do not make it hard to talk, only retainers do that!!!\nanswer_5: at the begining you feel really weird and it hurts a little bit but then you get used to themis not hard to talk at all you can talk normally\nanswer_6: I had them...they hurt for a bit when you first get them...feel tight.  then, they settle down.  However, they hurt each time you get them adjusted for a few days afterward.\nanswer_7: I just got my braces a few days ago. They felt like they were loose for the first two days, and it was really hard to eat. I got over speaking difficulties quickly, although the inside of your mouth gets kind of scratched and torn- don't fret, it heals quickly enough. After that you get calouses, so it doesn't hurt anymore. I think the pain felt depends on how old you are when you get them, what kind of treatment you are getting, and how severe your bite is. Lately, I've forgotten I'd had them on. You get used to them.\nanswer_8: Intially they are painful...also depending on your age and the condition your teeth are in. I have had my for almost 2 1/2 years and my first year was painful. Now I forget I have them...they are coming off soon and I can't wait! I recommend Aleve, its really helpful in relieving the pain.\nEXPERIENCE_GROUP: : yes yes yes. But not horribly painful. you get used to them and they become easier to talk with. the pain is only when they get tightened and until you get used to them. They give you wax to put around the spots that hurt you when your tongue rubs against the brackets and you won't feel the pain.They hurt for a bit when you first get them.They feel tight.Then, they settle down. However, they hurt each time you get them adjusted for a few days afterward. But,dont worry,you'll get used to the pain..they hurt when u first get them but then u get used 2 them really  easily. sometimes u like drool or spit or something unless u have a pallete expander u can talk normal. if u have 1 ur \"K's\" get all screwed up.Yes THEy HuRt WhEn YoU fIrSt GeT them but just for a few days...If you eat something hard and it gets stuck in them it mite hurt...and also after youy get them tightened..but they do not make it hard to talk, only retainers do that!.t the begining you feel really weird and it hurts a little bit but then you get used to themis not hard to talk at all you can talk normally.had them...they hurt for a bit when you first get them...feel tight.  then, they settle down.  However, they hurt each time you get them adjusted for a few days afterward..I just got my braces a few days ago. They felt like they were loose for the first two days, and it was really hard to eat. I got over speaking difficulties quickly, although the inside of your mouth gets kind of scratched and torn- don't fret, it heals quickly enough. After that you get calouses, so it doesn't hurt anymore. I think the pain felt depends on how old you are when you get them, what kind of treatment you are getting, and how severe your bite is. Lately, I've forgotten I'd had them on. You get used to them..Intially they are painful...also depending on your age and the condition your teeth are in. I have had my for almost 2 1/2 years and my first year was painful. Now I forget I have them...they are coming off soon and I can't wait! I recommend Aleve, its really helpful in relieving the pain.\n"
    }
]

def test_swasthyasetu_t5_endpoint():
    """Test the SwasthyaSetu T5 endpoint with braces data"""
    print("🏥 Testing SwasthyaSetu T5 Endpoint with Braces Data")
    print("=" * 60)
    
    print(f"📊 Data Size: {len(json.dumps(braces_data))} characters")
    print(f"📝 Questions: {len(braces_data)}")
    print(f"💬 Answers: {sum(len(item.get('answers', [])) for item in braces_data)}")
    
    try:
        start_time = time.time()
        
        print("\n🔄 Sending request to SwasthyaSetu T5 endpoint...")
        response = requests.post(
            "http://127.0.0.1:8000/summarize/swasthyasetu-t5",
            json=braces_data,
            headers={"Authorization": "Bearer test_token"},
            timeout=60
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            print("✅ SwasthyaSetu T5 endpoint successful!")
            data = response.json()
            
            print(f"\n⚡ Performance Results:")
            print(f"   Total request time: {total_time:.3f} seconds")
            print(f"   Processing time: {data.get('processing_info', {}).get('processing_time_seconds', 'N/A')} seconds")
            print(f"   Performance rating: {data.get('processing_info', {}).get('performance', 'N/A')}")
            print(f"   Model used: {data.get('processing_info', {}).get('model_used', 'N/A')}")
            print(f"   Model status: {data.get('processing_info', {}).get('model_status', 'N/A')}")
            
            print(f"\n📝 Generated Summaries:")
            print(f"   Patient Summary: {data.get('patient_summary', '')}")
            print(f"\n   Clinical Summary: {data.get('clinician_summary', '')}")
            
            print(f"\n📈 Quality Metrics:")
            print(f"   Faithfulness Score: {data.get('faithfulness_score', 'N/A')}")
            print(f"   Message: {data.get('message', 'N/A')}")
            
            print(f"\n🔍 Extracted Content Preview:")
            extracted = data.get('extracted_content', '')
            print(f"   {extracted[:200]}...")
            
        else:
            print(f"❌ SwasthyaSetu T5 endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ SwasthyaSetu T5 endpoint error: {e}")

def test_ultra_fast_comparison():
    """Test ultra-fast endpoint for comparison"""
    print("\n🔄 Testing Ultra-Fast Endpoint for Comparison")
    print("=" * 50)
    
    try:
        start_time = time.time()
        
        response = requests.post(
            "http://127.0.0.1:8000/summarize/ultra-fast",
            json=braces_data,
            headers={"Authorization": "Bearer test_token"},
            timeout=30
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            print("✅ Ultra-fast endpoint successful!")
            print(f"   Total request time: {total_time:.3f} seconds")
        else:
            print(f"❌ Ultra-fast endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Ultra-fast endpoint error: {e}")

def test_model_info():
    """Test model info endpoint"""
    print("\n📊 Testing Model Info Endpoint")
    print("=" * 40)
    
    try:
        response = requests.get("http://127.0.0.1:8000/model/info")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved successfully!")
            print(f"   Project: {data.get('project_name', 'N/A')}")
            print(f"   Description: {data.get('description', 'N/A')}")
            print(f"   Recommended endpoint: {data.get('recommended_endpoint', 'N/A')}")
            
            if 'swasthyasetu_t5' in data:
                t5_info = data['swasthyasetu_t5']
                print(f"   SwasthyaSetu T5: {t5_info.get('status', 'N/A')}")
                print(f"   T5 Model loaded: {t5_info.get('is_loaded', 'N/A')}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Model info error: {e}")

if __name__ == "__main__":
    print("🏥 SwasthyaSetu Medical Summarizer - Braces Data Test")
    print("=" * 70)
    
    # Wait for backend to be ready
    print("Waiting for backend to be ready...")
    time.sleep(5)
    
    # Test model info first
    test_model_info()
    
    # Test SwasthyaSetu T5 endpoint
    test_swasthyasetu_t5_endpoint()
    
    # Test ultra-fast endpoint for comparison
    test_ultra_fast_comparison()
    
    print("\n✨ Testing complete!")
    print("\n💡 Use /summarize/swasthyasetu-t5 for AI-powered summaries!")
    print("💡 Use /summarize/ultra-fast for instant results!")
