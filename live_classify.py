import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from src.cnn import BananaCNN

class LiveBananaClassifier:
    def __init__(self, model_path='banana_model.pth', frame_skip=10):
        """
        Initialize the live classifier
        
        Args:
            model_path: Path to saved model weights
            frame_skip: Process every Nth frame (e.g., 10 = process every 10th frame)
        """
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.current_prediction = None
        self.current_confidence = 0.0
        
        # Load the trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BananaCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Same preprocessing as training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['Unripe', 'Ripe']
        
    def preprocess_frame(self, frame):
        """Convert OpenCV frame to model input"""
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transformations
        input_tensor = self.transform(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        return input_batch
    
    def predict(self, frame):
        """Run inference on a single frame"""
        with torch.no_grad():
            input_batch = self.preprocess_frame(frame)
            outputs = self.model(input_batch)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item()
    
    def draw_prediction(self, frame):
        """Draw prediction text on frame"""
        if self.current_prediction is not None:
            label = self.class_names[self.current_prediction]
            confidence = self.current_confidence * 100
            
            # Choose color based on prediction
            color = (0, 255, 0) if self.current_prediction == 1 else (0, 165, 255)  # Green for ripe, Orange for unripe
            
            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            # Draw text
            cv2.putText(frame, f'Prediction: {label}', (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f'Confidence: {confidence:.1f}%', (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, 'Press Q to quit', (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main loop for live classification"""
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting live banana classifier...")
        print(f"Processing every {self.frame_skip} frames")
        print(f"Using device: {self.device}")
        print("Press Q to quit")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame if it's time (every Nth frame)
            if self.frame_count % self.frame_skip == 0:
                try:
                    prediction, confidence = self.predict(frame)
                    self.current_prediction = prediction
                    self.current_confidence = confidence
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            self.frame_count += 1
            
            # Draw prediction on frame
            annotated_frame = self.draw_prediction(frame)
            
            # Display
            cv2.imshow('Banana Ripeness Classifier', annotated_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Classification stopped")


if __name__ == "__main__":
    # You can adjust frame_skip here (higher = better performance, lower = more responsive)
    classifier = LiveBananaClassifier(model_path='banana_model.pth', frame_skip=10)
    classifier.run()