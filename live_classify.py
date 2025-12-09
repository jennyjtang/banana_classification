import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from src.cnn import ResNet_GAP_RD
from color_analyzer import BananaColorAnalyzer

class LiveBananaClassifier:
    def __init__(self, model_path='banana_model.pth', frame_skip=10, integrated_mode=True):
        """
        Initialize the live classifier with CNN and color analysis
        
        Args:
            model_path: Path to saved model weights
            frame_skip: Process every Nth frame
            integrated_mode: Use integrated CNN+color predictions (recommended)
        """
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.integrated_mode = integrated_mode
        
        # Results storage
        self.current_result = None
        self.detected_bbox = None  # Store detected banana bounding box
        
        # Load the trained ResNet model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet_GAP_RD()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize color analyzer
        self.color_analyzer = BananaColorAnalyzer()
        
        # Same preprocessing as training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['Unripe', 'Ripe']
    
    def detect_banana(self, frame):
        """
        Detect banana in frame using color-based segmentation and shape analysis
        Returns bounding box (x, y, w, h) or None if not detected
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for bananas (yellow/green/brown)
        # Yellow bananas
        lower_yellow = np.array([20, 40, 40])
        upper_yellow = np.array([40, 255, 255])
        
        # Green bananas
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Brown/darker bananas
        lower_brown = np.array([10, 40, 20])
        upper_brown = np.array([25, 255, 200])
        
        # Create masks
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Combine color masks
        color_mask = cv2.bitwise_or(mask_yellow, mask_green)
        color_mask = cv2.bitwise_or(color_mask, mask_brown)
        
        # Edge detection to help separate banana from background regardless of color
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate edges to create connected regions
        kernel_edge = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel_edge, iterations=2)
        
        # Combine color and edge information
        # This helps detect banana even against similar-colored backgrounds
        mask = cv2.bitwise_or(color_mask, edges_dilated)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Refine mask by keeping only regions that have significant color match
        # This prevents pure edge detections from non-banana objects
        mask = cv2.bitwise_and(mask, cv2.dilate(color_mask, kernel, iterations=3))
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find best banana candidate based on shape characteristics
        best_candidate = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < 1000:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio (bananas are elongated)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            
            # Calculate extent (ratio of contour area to bounding box area)
            # Bananas typically have extent between 0.5-0.8 due to their curved shape
            rect_area = w * h
            extent = area / (rect_area + 1e-5)
            
            # Approximate contour to detect curvature
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Calculate solidity (convexity)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-5)
            
            # Score based on banana-like characteristics:
            # - Elongated shape (aspect ratio 2.0-6.0)
            # - Curved/not rectangular (extent 0.4-0.8)
            # - Somewhat convex but not perfectly (solidity 0.7-0.95)
            # - Not too many sharp corners (fewer approximation points)
            
            score = 0
            
            # Aspect ratio scoring (prefer elongated shapes)
            if 2.0 <= aspect_ratio <= 6.0:
                score += 30
            elif 1.5 <= aspect_ratio < 2.0 or 6.0 < aspect_ratio <= 8.0:
                score += 15
            
            # Extent scoring (curved, not filling entire bbox)
            if 0.5 <= extent <= 0.8:
                score += 25
            elif 0.4 <= extent < 0.5 or 0.8 < extent <= 0.85:
                score += 10
            
            # Solidity scoring (somewhat convex)
            if 0.75 <= solidity <= 0.95:
                score += 20
            elif 0.65 <= solidity < 0.75:
                score += 10
            
            # Size scoring (larger is better)
            size_score = min(25, area / 500)
            score += size_score
            
            if score > best_score:
                best_score = score
                best_candidate = (x, y, w, h)
        
        # Only return detection if score is reasonable
        if best_candidate is None or best_score < 40:
            return None
        
        # Add padding around detection
        x, y, w, h = best_candidate
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        return (x, y, w, h)
        
    def preprocess_frame(self, frame):
        """Convert OpenCV frame to model input"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = self.transform(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        return input_batch
    
    def predict(self, frame):
        """Run CNN inference on a single frame"""
        with torch.no_grad():
            input_batch = self.preprocess_frame(frame)
            outputs = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return predicted.item(), confidence.item()
    
    def analyze_colors(self, frame):
        """Run color-based ripeness analysis"""
        return self.color_analyzer.analyze_frame(frame)
    
    def integrate_predictions(self, cnn_prediction, cnn_confidence, color_result):
        """Combine CNN and color analysis for final result"""
        
        if not color_result['detected']:
            return {
                'final_classification': self.class_names[cnn_prediction],
                'ripeness_bucket': 'Unknown',
                'ripeness_stage': 'No banana detected',
                'cnn_prediction': self.class_names[cnn_prediction],
                'cnn_confidence': cnn_confidence,
                'color_bucket': 'N/A',
                'agreement': False,
                'method': 'CNN only',
                'color_counts': None
            }
        
        color_bucket = color_result['bucket']
        color_stage = color_result['stage']
        ripeness_pct = color_result['ripeness_percentage']
        
        cnn_says_ripe = (cnn_prediction == 1)
        color_says_ripe = (ripeness_pct >= 30)
        
        # Integration logic
        if cnn_says_ripe and color_says_ripe:
            final_class = "Ripe"
            method = "CNN + Color"
        elif not cnn_says_ripe and not color_says_ripe:
            final_class = "Unripe"
            method = "CNN + Color"
        elif cnn_says_ripe and not color_says_ripe:
            final_class = "Ripe" if cnn_confidence > 0.8 else "Unripe"
            method = "CNN priority" if cnn_confidence > 0.8 else "Color priority"
        else:
            final_class = "Unripe" if cnn_confidence > 0.8 else "Ripe"
            method = "CNN priority" if cnn_confidence > 0.8 else "Color priority"
        
        return {
            'final_classification': final_class,
            'ripeness_bucket': color_bucket,
            'ripeness_stage': color_stage,
            'cnn_prediction': self.class_names[cnn_prediction],
            'cnn_confidence': cnn_confidence,
            'color_bucket': color_bucket,
            'color_stage': color_stage,
            'agreement': cnn_says_ripe == color_says_ripe,
            'method': method,
            'color_counts': color_result.get('color_counts')
        }
    
    def draw_results(self, frame):
        """Draw integrated results on frame"""
        
        # Draw bounding box if banana detected
        if self.detected_bbox is not None:
            x, y, w, h = self.detected_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Banana Detected', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Banana Detected - Using Full Frame', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if self.current_result is None:
            return frame
        
        result = self.current_result
        
        # Create semi-transparent overlay panel
        overlay = frame.copy()
        panel_height = 220
        cv2.rectangle(overlay, (10, 10), (500, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        y_offset = 45
        
        # Final classification (large and prominent)
        final_class = result['final_classification']
        color = (0, 255, 0) if final_class == "Ripe" else (0, 165, 255)
        cv2.putText(frame, f'Classification: {final_class}', (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        y_offset += 45
        
        # Stage description (more detailed than binary classification)
        stage = result.get('ripeness_stage', '')
        # Extract simple label from stage (e.g., "Green (Unripe)" -> "Green")
        stage_label = stage.split('(')[0].strip() if stage else 'N/A'
        stage_color = (0, 255, 255) if stage != 'N/A' else (128, 128, 128)
        cv2.putText(frame, f'Stage: {stage_label}', (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage_color, 2)
        y_offset += 35
        
        # Full stage description with bucket
        bucket = result.get('ripeness_bucket', '')
        full_stage = f'{stage_label} ({bucket})' if bucket and bucket != 'N/A' else stage
        cv2.putText(frame, full_stage[:40], (20, y_offset),  # Truncate if too long
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 40
        
        # Show method and agreement
        method = result.get('method', 'Unknown')
        agreement = result.get('agreement', False)
        agreement_text = "Agree" if agreement else "Disagree"
        agreement_color = (0, 255, 0) if agreement else (0, 140, 255)
        
        cv2.putText(frame, f'Method: {method}', (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        
        cv2.putText(frame, f'CNN & Color: {agreement_text}', (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, agreement_color, 1)
        y_offset += 30
        
        # Color distribution bar (if available)
        counts = result.get('color_counts')
        if counts and counts['total'] > 0:
            total = counts['total']
            bar_y = y_offset
            bar_x = 20
            bar_width = 450
            bar_height = 20
            
            green_width = int((counts['green'] / total) * bar_width)
            yellow_width = int((counts['yellow'] / total) * bar_width)
            brown_width = int((counts['brown'] / total) * bar_width)
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), 
                        (bar_x + bar_width, bar_y + bar_height), 
                        (50, 50, 50), -1)
            
            # Draw segments
            x = bar_x
            if green_width > 0:
                cv2.rectangle(frame, (x, bar_y), 
                            (x + green_width, bar_y + bar_height), 
                            (0, 200, 0), -1)
                x += green_width
            
            if yellow_width > 0:
                cv2.rectangle(frame, (x, bar_y), 
                            (x + yellow_width, bar_y + bar_height), 
                            (0, 255, 255), -1)
                x += yellow_width
            
            if brown_width > 0:
                cv2.rectangle(frame, (x, bar_y), 
                            (x + brown_width, bar_y + bar_height), 
                            (19, 69, 139), -1)
        
        # Instructions at bottom
        cv2.putText(frame, 'Press Q to quit | I to toggle integrated mode', 
                   (10, frame.shape[0] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    

    def run(self):
        """Main loop for live classification with integrated predictions"""

        # MacBook Pro screen is typically 1440x900 or 1680x1050, so half would be around 720x450 or 840x525
        # WINDOW_WIDTH = 1440 
        # WINDOW_HEIGHT = 900
        # WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

        # cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Camera Feed', WINDOW_SIZE[0], WINDOW_SIZE[1])

        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("=" * 60)
        print("INTEGRATED BANANA RIPENESS CLASSIFIER")
        print("=" * 60)
        print(f"CNN Model: Loaded")
        print(f"Color Analysis: Enabled")
        print(f"Integrated Mode: {'ON' if self.integrated_mode else 'OFF'}")
        print(f"Processing every {self.frame_skip} frames")
        print(f"Device: {self.device}")
        print("\nControls:")
        print("  Q - Quit")
        print("  I - Toggle integrated mode (combine CNN + color)")
        print("=" * 60)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame periodically
            if self.frame_count % self.frame_skip == 0:
                try:
                    # Detect banana in frame
                    self.detected_bbox = self.detect_banana(frame)
                    
                    # If banana detected, crop to that region for analysis
                    if self.detected_bbox is not None:
                        x, y, w, h = self.detected_bbox
                        analysis_frame = frame[y:y+h, x:x+w]
                    else:
                        # Use full frame if no banana detected
                        analysis_frame = frame
                    
                    # CNN prediction on detected region
                    cnn_pred, cnn_conf = self.predict(analysis_frame)
                    
                    # Color analysis on detected region
                    color_result = self.analyze_colors(analysis_frame)
                    
                    # Integrate predictions
                    if self.integrated_mode:
                        self.current_result = self.integrate_predictions(
                            cnn_pred, cnn_conf, color_result
                        )
                    else:
                        # Just show CNN results
                        self.current_result = {
                            'final_classification': self.class_names[cnn_pred],
                            'ripeness_bucket': 'N/A',
                            'ripeness_stage': 'Integrated mode OFF',
                            'cnn_prediction': self.class_names[cnn_pred],
                            'cnn_confidence': cnn_conf,
                            'color_bucket': 'N/A',
                            'agreement': False,
                            'method': 'CNN only',
                            'color_counts': None
                        }
                        
                except Exception as e:
                    print(f"Analysis error: {e}")
            
            self.frame_count += 1
            
            # Draw results
            annotated_frame = self.draw_results(frame)
            
            # Display
            cv2.imshow('Integrated Banana Classifier', annotated_frame)
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                self.integrated_mode = not self.integrated_mode
                status = "enabled" if self.integrated_mode else "disabled"
                print(f"Integrated mode {status}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nClassification stopped")


if __name__ == "__main__":
    # Use integrated mode by default (combines CNN + color analysis)
    classifier = LiveBananaClassifier(
        model_path='banana_model.pth', 
        frame_skip=30,
        integrated_mode=True
    )
    classifier.run()