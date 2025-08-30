import requests
import json
import time
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"  # Using port 8001 for our running instance
TEST_USER = {
    "username": "test_user_" + str(int(time.time())),
    "password": "test_password",
    "email": f"test_{int(time.time())}@example.com"
}

class TestMedicalSummarizerAPI:
    """Comprehensive test suite for Medical Summarizer API"""
    
    def __init__(self):
        self.token = None
        self.session = requests.Session()
    
    def test_health_check(self):
        """Test basic API health check"""
        response = self.session.get(f"{API_BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "Medical Summarizer API" in data["message"]
        print("✓ Health check passed")
    
    def test_model_info(self):
        """Test model information endpoint"""
        response = self.session.get(f"{API_BASE_URL}/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "Perspective-Aware" in data["model_type"]
        assert data["model_loaded"] == True
        print("✓ Model info endpoint working")
    
    def test_user_registration(self):
        """Test user registration"""
        response = self.session.post(
            f"{API_BASE_URL}/auth/register",
            json=TEST_USER
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "User registered successfully"
        print("✓ User registration successful")
    
    def test_user_login(self):
        """Test user login and token generation"""
        response = self.session.post(
            f"{API_BASE_URL}/auth/login",
            json={
                "username": TEST_USER["username"],
                "password": TEST_USER["password"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        
        # Store token for subsequent tests
        self.token = data["access_token"]
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        print("✓ User login successful")
    
    def test_summarization_patient_mode(self):
        """Test summarization in patient mode"""
        if not self.token:
            self.test_user_login()
        
        test_text = """
        Question: What is diabetes and how should I manage my blood sugar levels?
        
        Answer: Diabetes is a condition where your blood sugar levels are too high. 
        You can manage it by eating healthy foods, exercising regularly, taking your 
        medications as prescribed, and monitoring your blood sugar levels daily.
        """
        
        response = self.session.post(
            f"{API_BASE_URL}/summarize",
            json={
                "text": test_text,
                "mode": "patient"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "patient_summary" in data
        assert data["patient_summary"] is not None
        assert "disclaimer" in data
        print("✓ Patient mode summarization working")
    
    def test_summarization_clinician_mode(self):
        """Test summarization in clinician mode"""
        if not self.token:
            self.test_user_login()
        
        test_text = """
        Question: What are the treatment options for Type 2 diabetes mellitus?
        
        Answer: Treatment includes lifestyle modifications, metformin as first-line therapy,
        and additional medications like sulfonylureas or insulin if needed. Regular HbA1c
        monitoring is essential for glycemic control assessment.
        """
        
        response = self.session.post(
            f"{API_BASE_URL}/summarize",
            json={
                "text": test_text,
                "mode": "clinician"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "clinician_summary" in data
        assert data["clinician_summary"] is not None
        print("✓ Clinician mode summarization working")
    
    def test_summarization_both_modes(self):
        """Test summarization in both modes"""
        if not self.token:
            self.test_user_login()
        
        test_text = """
        Question: I have high blood pressure. What should I do?
        
        Answer: High blood pressure can be managed through lifestyle changes like reducing 
        salt intake, regular exercise, maintaining healthy weight, and taking prescribed 
        medications. Regular monitoring is important to prevent complications.
        """
        
        response = self.session.post(
            f"{API_BASE_URL}/summarize",
            json={
                "text": test_text,
                "mode": "both"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "patient_summary" in data
        assert "clinician_summary" in data
        assert data["patient_summary"] is not None
        assert data["clinician_summary"] is not None
        assert "faithfulness_score" in data
        assert "provenance_scores" in data
        print("✓ Both modes summarization working")
    
    def test_feedback_submission(self):
        """Test feedback submission"""
        if not self.token:
            self.test_user_login()
        
        response = self.session.post(
            f"{API_BASE_URL}/feedback",
            json={
                "summary_id": "test_summary_1",
                "rating": 4,
                "comments": "Good summary, very helpful!"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Feedback submitted successfully"
        print("✓ Feedback submission working")
    
    def test_history_retrieval(self):
        """Test history retrieval"""
        if not self.token:
            self.test_user_login()
        
        response = self.session.get(f"{API_BASE_URL}/history")
        assert response.status_code == 200
        data = response.json()
        assert "summaries" in data
        assert isinstance(data["summaries"], list)
        print("✓ History retrieval working")
    
    def test_unauthorized_access(self):
        """Test unauthorized access protection"""
        # Remove authorization header
        headers = self.session.headers.copy()
        if "Authorization" in self.session.headers:
            del self.session.headers["Authorization"]
        
        response = self.session.post(
            f"{API_BASE_URL}/summarize",
            json={
                "text": "Test text",
                "mode": "patient"
            }
        )
        assert response.status_code == 401
        
        # Restore headers
        self.session.headers = headers
        print("✓ Unauthorized access protection working")
    
    def test_invalid_mode(self):
        """Test invalid summarization mode"""
        if not self.token:
            self.test_user_login()
        
        response = self.session.post(
            f"{API_BASE_URL}/summarize",
            json={
                "text": "Test text",
                "mode": "invalid_mode"
            }
        )
        # Should still work as backend handles invalid modes gracefully
        assert response.status_code in [200, 400]
        print("✓ Invalid mode handling working")
    
    def test_empty_text_summarization(self):
        """Test summarization with empty text"""
        if not self.token:
            self.test_user_login()
        
        response = self.session.post(
            f"{API_BASE_URL}/summarize",
            json={
                "text": "",
                "mode": "patient"
            }
        )
        # Should handle empty text gracefully
        assert response.status_code in [200, 400]
        print("✓ Empty text handling working")
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🧪 Running Medical Summarizer API Tests")
        print("=" * 50)
        
        try:
            self.test_health_check()
            self.test_model_info()
            self.test_user_registration()
            self.test_user_login()
            self.test_summarization_patient_mode()
            self.test_summarization_clinician_mode()
            self.test_summarization_both_modes()
            self.test_feedback_submission()
            self.test_history_retrieval()
            self.test_unauthorized_access()
            self.test_invalid_mode()
            self.test_empty_text_summarization()
            
            print("\n" + "=" * 50)
            print("🎉 All tests passed successfully!")
            print("✅ Medical Summarizer API is working correctly")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise
        
        except AssertionError as e:
            print(f"\n❌ Assertion failed: {e}")
            raise

def run_performance_tests():
    """Run performance benchmarks"""
    print("\n🚀 Running Performance Tests")
    print("=" * 30)
    
    api_tester = TestMedicalSummarizerAPI()
    api_tester.test_user_registration()
    api_tester.test_user_login()
    
    # Test response times
    test_text = """
    Question: What are the symptoms of heart disease?
    Answer: Common symptoms include chest pain, shortness of breath, 
    fatigue, and irregular heartbeat. Early detection is important.
    """
    
    # Measure summarization time
    start_time = time.time()
    response = api_tester.session.post(
        f"{API_BASE_URL}/summarize",
        json={"text": test_text, "mode": "both"}
    )
    end_time = time.time()
    
    response_time = end_time - start_time
    print(f"✓ Summarization response time: {response_time:.2f} seconds")
    
    if response_time < 5.0:
        print("✅ Performance: Excellent (< 5s)")
    elif response_time < 10.0:
        print("⚠️ Performance: Good (< 10s)")
    else:
        print("❌ Performance: Needs improvement (> 10s)")

if __name__ == "__main__":
    # Run API tests
    tester = TestMedicalSummarizerAPI()
    tester.run_all_tests()
    
    # Run performance tests
    run_performance_tests()

