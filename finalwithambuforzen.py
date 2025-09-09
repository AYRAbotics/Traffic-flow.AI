import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
from scipy.spatial.distance import cdist
import os

class AerialTrafficAnalyzer:
    def __init__(self, video_path, model_path="yolov8x.pt", ambulance_refs=None):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.vehicle_counts = defaultdict(int)
        self.vehicle_trajectories = defaultdict(list)
        self.traffic_violations = defaultdict(list)
        self.congestion_points = []
        self.lane_vehicle_counts = defaultdict(lambda: defaultdict(int))
        self.ambulance_detections = []
        
        # Initialize SIFT and matcher first
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher()
        
        # Initialize ambulance detection
        self.ambulance_refs = ambulance_refs or ["1.png", "2.png"]
        self.ambulance_templates = self._load_ambulance_templates()

        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        self.lane_polygons = {
            "lane 1": np.array([
                (2010, 1608), (2436, 1095), (3516, 2094), (2265, 2124), (2010, 1626)
            ]),
            "lane 2": np.array([
                (1620, 657), (1926, 1065), (1392, 1488), (432, 1989), (252, 1560), (1596, 672)
            ]),
            "lane 3": np.array([
                (1545, 333), (1764, 318), (2151, 627), (1956, 768), (1554, 354)
            ]),
            "lane 4": np.array([
                (2226, 759), (2658, 318), (2832, 402), (2565, 1011), (2235, 765)
            ])
        }

    def _load_ambulance_templates(self):
        """Load and process ambulance reference images"""
        templates = []
        for ref_path in self.ambulance_refs:
            if os.path.exists(ref_path):
                img = cv2.imread(ref_path)
                if img is not None:
                    # Convert to grayscale and extract features
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    keypoints, descriptors = self.sift.detectAndCompute(gray, None)
                    if descriptors is not None:
                        templates.append({
                            'image': img,
                            'gray': gray,
                            'keypoints': keypoints,
                            'descriptors': descriptors
                        })
                        print(f"Loaded ambulance template: {ref_path}")
                    else:
                        print(f"Warning: No features found in {ref_path}")
                else:
                    print(f"Warning: Could not load {ref_path}")
            else:
                print(f"Warning: Reference image {ref_path} not found")
        return templates

    def _detect_ambulance_in_roi(self, frame, x1, y1, x2, y2):
        """Detect ambulance in a specific region of interest"""
        if not self.ambulance_templates:
            return False
            
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
            
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Extract features from ROI
        roi_kp, roi_desc = self.sift.detectAndCompute(roi_gray, None)
        if roi_desc is None:
            return False
        
        # Compare with each ambulance template
        for template in self.ambulance_templates:
            try:
                # Match features
                matches = self.bf_matcher.knnMatch(template['descriptors'], roi_desc, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                # If enough good matches found, consider it an ambulance
                if len(good_matches) > 15:  # Threshold for ambulance detection
                    return True
                    
            except Exception as e:
                print(f"Error in ambulance detection: {e}")
                continue
                
        return False

    def _get_lane(self, center):
        for lane_name, polygon in self.lane_polygons.items():
            if cv2.pointPolygonTest(polygon, tuple(center), False) >= 0:
                return lane_name
        return None

    def process_frame(self, frame):
        results = self.model(frame)[0]
        self.vehicle_counts.clear()
        self.lane_vehicle_counts.clear()
        self.ambulance_detections.clear()
        processed_frame = frame.copy()

        # Initialize y_pos at the start of the method
        y_pos = 40  # Start position for text overlay

        boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.array([])
        classes = results.boxes.cls.cpu().numpy() if len(results.boxes) > 0 else np.array([])
        confidences = results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else np.array([])
        track_ids = getattr(results.boxes, 'id', None)

        if track_ids is not None:
            track_ids = track_ids.cpu().numpy()
        else:
            track_ids = np.arange(len(boxes))

        for box, cls_id, conf, track_id in zip(boxes, classes, confidences, track_ids):
            if int(cls_id) in self.vehicle_classes:
                self.vehicle_counts[int(cls_id)] += 1

                x1, y1, x2, y2 = map(int, box)
                center = [(x1 + x2) // 2, (y1 + y2) // 2]

                lane = self._get_lane(center)
                if lane:
                    self.lane_vehicle_counts[lane][int(cls_id)] += 1

                # Check for ambulance in this vehicle detection
                is_ambulance = self._detect_ambulance_in_roi(frame, x1, y1, x2, y2)
                
                if is_ambulance:
                    # Draw red box for ambulance
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = "AMBULANCE"
                    cv2.putText(processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    
                    # Store ambulance detection
                    if lane:
                        self.ambulance_detections.append(f"Emergency vehicle in {lane}")
                    else:
                        self.ambulance_detections.append("Emergency vehicle detected")
                else:
                    # Regular vehicle - green box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    label = f"{self.vehicle_classes[int(cls_id)]}"
                    cv2.putText(processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 200, 0), 2)

                self.vehicle_trajectories[track_id].append(center)

        self._detect_violations()
        self._analyze_congestion(boxes)
        self._predict_trajectories()

        # Draw total vehicles count with larger font
        text = f"Total Vehicles: {sum(self.vehicle_counts.values())}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.rectangle(processed_frame, (8, y_pos - text_height - 8), (10 + text_width + 8, y_pos + 8), (255, 255, 255), -1)
        cv2.putText(processed_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        # Draw ambulance alerts
        y_pos += 50
        for alert in self.ambulance_detections:
            alert_text = f"ALERT: {alert}"
            (alert_width, alert_height), alert_baseline = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(processed_frame, (8, y_pos - alert_height - 8), (10 + alert_width + 8, y_pos + 8), (0, 0, 255), -1)
            cv2.putText(processed_frame, alert_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            y_pos += 40

        # Draw lane counts with increased spacing and larger font
        y_pos += 10
        for lane, cls_dict in self.lane_vehicle_counts.items():
            lane_text = f"{lane.upper()}: " + ', '.join(
                f"{self.vehicle_classes[cls_id]}: {count}" for cls_id, count in cls_dict.items())
            (lane_text_width, lane_text_height), lane_baseline = cv2.getTextSize(lane_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(processed_frame, (8, y_pos - lane_text_height - 8), (10 + lane_text_width + 8, y_pos + 8), (255, 255, 255), -1)
            cv2.putText(processed_frame, lane_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            y_pos += 40

        return processed_frame

    def _detect_violations(self):
        for tracker_id, trajectory in self.vehicle_trajectories.items():
            if len(trajectory) >= 2:
                positions = np.array(trajectory[-2:])
                dist = np.linalg.norm(positions[1] - positions[0])
                if dist > 50:
                    if "Speeding" not in self.traffic_violations[tracker_id]:
                        self.traffic_violations[tracker_id].append("Speeding")

    def _analyze_congestion(self, boxes):
        self.congestion_points = []
        if len(boxes) > 1:
            centroids = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
            distances = cdist(centroids, centroids)
            neighbor_counts = (distances < 100).sum(axis=1) - 1
            congestion_indices = np.where(neighbor_counts > 5)[0]
            if len(congestion_indices) > 0:
                self.congestion_points = centroids[congestion_indices]

    def _predict_trajectories(self):
        pass  # Expand later

    def _generate_report(self):
        report = {
            "Total Vehicles": sum(self.vehicle_counts.values()),
            "Vehicle Types": dict(self.vehicle_counts),
            "Traffic Violations": sum(len(v) for v in self.traffic_violations.values()),
            "Congestion Points": len(self.congestion_points),
            "Emergency Vehicles Detected": len(self.ambulance_detections)
        }
        report_path = self.video_path.rsplit('.', 1)[0] + "_traffic_report.csv"
        pd.DataFrame([report]).to_csv(report_path, index=False)
        print(f"Report saved to {report_path}")

    def run_and_save(self, output_path="resultlanecount.mp4"):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {self.video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        print("Starting video processing...")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed = self.process_frame(frame)
            out.write(processed)
            
            # Print ambulance detections to console
            if self.ambulance_detections:
                print(f"Frame {frame_count}: {', '.join(self.ambulance_detections)}")
            
            cv2.imshow("Traffic Analysis", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Early exit requested.")
                break
            
            frame_count += 1

        self._generate_report()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video processing complete. Output saved to {output_path}")

def main():
    video_path = "input.MOV"  # Replace with your actual video file path
    output_path = "annotated_output.mp4"
    
    # Specify ambulance reference images
    ambulance_refs = ["1.png", "2.png"]
    
    analyzer = AerialTrafficAnalyzer(video_path, ambulance_refs=ambulance_refs)
    analyzer.run_and_save(output_path)

if __name__ == "__main__":
    main()