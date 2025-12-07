"""
Test script for color-based ripeness analysis.
Run this to test the color analyzer on static images or live feed.
"""

import cv2
import os
from color_analyzer import BananaColorAnalyzer

def test_on_image(image_path, analyzer):
    """Test color analysis on a single image"""
    print(f"\nAnalyzing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Error: Could not load image")
        return
    
    # Analyze
    result = analyzer.analyze_frame(image)
    
    # Print results
    if result['detected']:
        print(f"  Detected: Yes")
        print(f"  Ripeness: {result['ripeness_percentage']:.1f}%")
        print(f"  Stage: {result['stage']}")
        
        if result['color_counts']:
            counts = result['color_counts']
            total = counts['total']
            print(f"  Color breakdown:")
            print(f"    Green:  {counts['green']:6d} ({counts['green']/total*100:5.1f}%)")
            print(f"    Yellow: {counts['yellow']:6d} ({counts['yellow']/total*100:5.1f}%)")
            print(f"    Brown:  {counts['brown']:6d} ({counts['brown']/total*100:5.1f}%)")
        
        # Show visualization
        annotated = analyzer.visualize_analysis(image, result)
        
        # Resize for display if too large
        max_height = 800
        if annotated.shape[0] > max_height:
            scale = max_height / annotated.shape[0]
            new_width = int(annotated.shape[1] * scale)
            annotated = cv2.resize(annotated, (new_width, max_height))
        
        cv2.imshow('Color Analysis', annotated)
        print("  Press any key to continue...")
        cv2.waitKey(0)
    else:
        print(f"  Detected: No (banana not found)")

def test_on_directory(directory, analyzer):
    """Test on all images in a directory"""
    print(f"\nScanning directory: {directory}")
    
    image_files = [f for f in os.listdir(directory) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"Found {len(image_files)} images")
    
    for filename in sorted(image_files):
        image_path = os.path.join(directory, filename)
        test_on_image(image_path, analyzer)
    
    cv2.destroyAllWindows()

def test_live_feed(analyzer):
    """Test on live webcam feed"""
    print("\nStarting live color analysis test...")
    print("Press Q to quit")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze frame
        result = analyzer.analyze_frame(frame)
        
        # Draw results
        annotated = frame.copy()
        
        if result['detected']:
            # Visualize
            annotated = analyzer.visualize_analysis(frame, result)
            
            # Add text overlay
            ripeness = result['ripeness_percentage']
            stage = result['stage']
            
            cv2.putText(annotated, f"Ripeness: {ripeness:.1f}%", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f"Stage: {stage}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(annotated, "No banana detected", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Live Color Analysis Test', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("=" * 70)
    print("BANANA COLOR ANALYSIS TEST")
    print("=" * 70)
    
    analyzer = BananaColorAnalyzer()
    
    print("\nChoose test mode:")
    print("1. Test on single image")
    print("2. Test on image directory")
    print("3. Test on live webcam feed")
    print("4. Quick test on training data")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        image_path = input("Enter image path: ").strip()
        test_on_image(image_path, analyzer)
    
    elif choice == '2':
        directory = input("Enter directory path: ").strip()
        test_on_directory(directory, analyzer)
    
    elif choice == '3':
        test_live_feed(analyzer)
    
    elif choice == '4':
        # Test on some training images
        data_dir = "data_banana/train/images"
        if os.path.exists(data_dir):
            test_on_directory(data_dir, analyzer)
        else:
            print(f"Error: Directory not found: {data_dir}")
    
    else:
        print("Invalid choice")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()