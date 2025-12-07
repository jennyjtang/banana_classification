import cv2
import numpy as np

class BananaColorAnalyzer:
    def __init__(self):
        """
        Initialize color thresholds for banana ripeness detection in HSV color space.
        HSV is better than RGB for color-based detection because:
        - H (Hue): Color type (green, yellow, brown)
        - S (Saturation): Color intensity
        - V (Value): Brightness
        """
        
        # HSV color ranges for different ripeness stages
        # Format: [H_min, S_min, V_min], [H_max, S_max, V_max]
        
        # Green (unripe banana) - Hue ~35-85
        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([85, 255, 255])
        
        # Yellow (ripe banana) - Hue ~15-35
        self.yellow_lower = np.array([15, 40, 100])
        self.yellow_upper = np.array([35, 255, 255])
        
        # Brown (overripe/spots) - Hue ~5-15, lower saturation
        self.brown_lower = np.array([5, 40, 20])
        self.brown_upper = np.array([25, 255, 200])
        
        # General banana mask (to separate from background)
        # Captures all banana colors (green + yellow + brown)
        self.banana_lower = np.array([5, 30, 20])
        self.banana_upper = np.array([85, 255, 255])
    
    def extract_banana_mask(self, image):
        """
        Extract banana region from background using color masking.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            mask: Binary mask where banana pixels are white (255)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for banana colors
        mask = cv2.inRange(hsv, self.banana_lower, self.banana_upper)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        
        # Find largest contour (assume it's the banana)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Keep only the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        return mask
    
    def count_color_pixels(self, image, mask):
        """
        Count pixels of each color (green, yellow, brown) within the banana mask.
        
        Args:
            image: BGR image from OpenCV
            mask: Binary mask of banana region
            
        Returns:
            dict: Pixel counts for each color category
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply banana mask to focus only on banana pixels
        hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask)
        
        # Count pixels in each color range
        green_mask = cv2.inRange(hsv_masked, self.green_lower, self.green_upper)
        yellow_mask = cv2.inRange(hsv_masked, self.yellow_lower, self.yellow_upper)
        brown_mask = cv2.inRange(hsv_masked, self.brown_lower, self.brown_upper)
        
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        brown_pixels = cv2.countNonZero(brown_mask)
        total_banana_pixels = cv2.countNonZero(mask)
        
        return {
            'green': green_pixels,
            'yellow': yellow_pixels,
            'brown': brown_pixels,
            'total': total_banana_pixels
        }
    
    def calculate_ripeness_percentage(self, color_counts):
        """
        Calculate ripeness percentage based on color distribution.
        
        Logic:
        - More green = less ripe (0-40%)
        - More yellow = ripe (40-80%)
        - Brown spots = very ripe/overripe (80-100%)
        
        Args:
            color_counts: Dictionary with pixel counts
            
        Returns:
            float: Ripeness percentage (0-100)
        """
        green = color_counts['green']
        yellow = color_counts['yellow']
        brown = color_counts['brown']
        total = color_counts['total']
        
        if total == 0:
            return 0.0
        
        # Calculate ratios
        green_ratio = green / total
        yellow_ratio = yellow / total
        brown_ratio = brown / total
        
        # Ripeness formula (weighted calculation)
        # Green contributes 0-30% ripeness
        # Yellow contributes 30-80% ripeness
        # Brown contributes 80-100% ripeness
        
        ripeness = 0.0
        
        # Green stage (0-40% ripe)
        if green_ratio > 0:
            ripeness += green_ratio * 20  # Max 20% from green
        
        # Yellow stage (adds to ripeness)
        if yellow_ratio > 0:
            ripeness += yellow_ratio * 60  # Can contribute up to 60%
        
        # Brown stage (indicates high ripeness)
        if brown_ratio > 0:
            ripeness += brown_ratio * 100  # Brown spots indicate advanced ripeness
        
        # Normalize: if mostly green, cap at lower ripeness
        if green_ratio > 0.6:
            ripeness = min(ripeness, 35)
        
        # If mostly yellow, ensure it's in ripe range
        if yellow_ratio > 0.6:
            ripeness = max(ripeness, 50)
        
        # If significant brown, ensure high ripeness
        if brown_ratio > 0.2:
            ripeness = max(ripeness, 75)
        
        return min(100.0, max(0.0, ripeness))
    
    def get_ripeness_stage(self, ripeness_percentage):
        """
        Convert ripeness percentage to descriptive stage with bucket range.
        
        Args:
            ripeness_percentage: Float 0-100
            
        Returns:
            tuple: (stage description, bucket range string)
        """
        if ripeness_percentage < 30:
            return "Green (Unripe)", "0-30%"
        elif ripeness_percentage < 50:
            return "Yellow-Green (Early Ripening)", "30-50%"
        elif ripeness_percentage < 70:
            return "Bright Yellow (Optimal)", "50-70%"
        elif ripeness_percentage < 90:
            return "Yellow with Brown Spots (Very Ripe)", "70-90%"
        else:
            return "Heavy Brown (Overripe)", "90-100%"
    
    def analyze_frame(self, frame):
        """
        Complete analysis of a frame to get ripeness percentage.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            dict: Analysis results including ripeness percentage, stage, and mask
        """
        try:
            # Extract banana region
            banana_mask = self.extract_banana_mask(frame)
            
            # Check if banana was detected
            if cv2.countNonZero(banana_mask) < 100:  # Minimum pixel threshold
                return {
                    'detected': False,
                    'ripeness_percentage': 0.0,
                    'stage': 'No banana detected',
                    'bucket': 'N/A',
                    'color_counts': None,
                    'mask': banana_mask
                }
            
            # Count color pixels
            color_counts = self.count_color_pixels(frame, banana_mask)
            
            # Calculate ripeness
            ripeness_percentage = self.calculate_ripeness_percentage(color_counts)
            stage, bucket = self.get_ripeness_stage(ripeness_percentage)
            
            return {
                'detected': True,
                'ripeness_percentage': ripeness_percentage,
                'stage': stage,
                'bucket': bucket,
                'color_counts': color_counts,
                'mask': banana_mask
            }
        
        except Exception as e:
            print(f"Error in color analysis: {e}")
            return {
                'detected': False,
                'ripeness_percentage': 0.0,
                'stage': 'Analysis error',
                'bucket': 'N/A',
                'color_counts': None,
                'mask': None
            }
    
    def visualize_analysis(self, frame, analysis_result):
        """
        Create a visualization showing the banana mask and color breakdown.
        
        Args:
            frame: Original BGR image
            analysis_result: Result from analyze_frame()
            
        Returns:
            annotated_frame: Frame with visual overlay
        """
        annotated = frame.copy()
        
        if not analysis_result['detected']:
            return annotated
        
        mask = analysis_result['mask']
        
        # Draw banana outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
        
        # Draw color distribution bar
        if analysis_result['color_counts']:
            counts = analysis_result['color_counts']
            total = counts['total']
            
            if total > 0:
                bar_height = 30
                bar_y = frame.shape[0] - 50
                bar_x_start = 20
                bar_width = 300
                
                # Calculate segment widths
                green_width = int((counts['green'] / total) * bar_width)
                yellow_width = int((counts['yellow'] / total) * bar_width)
                brown_width = int((counts['brown'] / total) * bar_width)
                
                # Draw background
                cv2.rectangle(annotated, (bar_x_start, bar_y), 
                            (bar_x_start + bar_width, bar_y + bar_height), 
                            (50, 50, 50), -1)
                
                # Draw color segments
                x_offset = bar_x_start
                
                if green_width > 0:
                    cv2.rectangle(annotated, (x_offset, bar_y), 
                                (x_offset + green_width, bar_y + bar_height), 
                                (0, 255, 0), -1)
                    x_offset += green_width
                
                if yellow_width > 0:
                    cv2.rectangle(annotated, (x_offset, bar_y), 
                                (x_offset + yellow_width, bar_y + bar_height), 
                                (0, 255, 255), -1)
                    x_offset += yellow_width
                
                if brown_width > 0:
                    cv2.rectangle(annotated, (x_offset, bar_y), 
                                (x_offset + brown_width, bar_y + bar_height), 
                                (19, 69, 139), -1)
                
                # Label
                cv2.putText(annotated, "Color Distribution", (bar_x_start, bar_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated