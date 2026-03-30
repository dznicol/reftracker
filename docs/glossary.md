# RefTracker — Technical Glossary

## Computer Vision & Detection

The core tech that finds and follows the referee in video footage.

| Term | What It Means |
|------|---------------|
| CV | Computer Vision — the field of AI that teaches computers to interpret images and video. RefTracker uses CV to detect people in each video frame and figure out which one is the referee. |
| YOLOv8 | "You Only Look Once" version 8. A real-time object detection model by Ultralytics. We use YOLOv8n (nano — the smallest, fastest variant). Pre-trained on COCO, detects 80 object classes including "person" (class 0). Free and open source. |
| YOLOv8n | The nano variant of YOLOv8. Smallest model size (~6MB), fastest inference speed. Trades a tiny bit of accuracy for real-time performance — perfect for processing video on a laptop. |
| COCO | "Common Objects in Context." A massive labelled image dataset (330k+ images, 80 categories) used to train models like YOLO. We use the pre-trained weights as-is — no custom training required. |
| Bounding Box | The rectangle drawn around a detected person, defined by four coordinates (x1, y1, x2, y2). We crop the bounding box to examine what the person is wearing. |
| BoTSORT | A multi-object tracking algorithm with ReID (re-identification) support. Assigns each detected person a persistent track ID across frames using both motion prediction and appearance features. |
| CLIP-ReID | CLIP-based re-identification model. Uses CLIP's semantic understanding of appearance to generate embeddings that distinguish people by what they look like, not just where they are. More robust than surveillance-trained ReID models for sports footage. |
| Track ID | A unique number assigned by the tracker to each person it follows. During calibration, we figure out which track ID belongs to the referee based on kit colour. |
| Calibration | The first ~60 frames where we scan every detected person's shirt colour to identify the referee's track ID before processing the rest of the video. |
| Track Lock | When the system is confidently following the referee's track ID frame-to-frame. Track lock is lost when the tracker reassigns the ref a new ID (due to occlusion, detection dropout, etc.). |
| Re-ID | Re-identification. When track lock is lost, the system scores all nearby people by colour to find the ref again and lock onto their new track ID. |
| Proximity Gate | During re-ID, we only consider candidates within a search radius of where we last saw the ref. A person can only run so far in a few frames, so this prevents locking onto the linesman on the far side of the pitch. |
| mAP | Mean Average Precision — the standard accuracy metric for object detection. YOLOv8n scores 37.3% on COCO; for person detection in sports video, even 37% is more than adequate. |

## Colour Detection

How we tell the referee apart from the players by what they're wearing.

| Term | What It Means |
|------|---------------|
| HSV | Hue, Saturation, Value — a colour model much better than RGB for filtering by colour. Hue = the colour itself (0–180 in OpenCV), Saturation = how vivid, Value = how bright. We use it to distinguish "kit green" from "grass green." |
| Grass Masking | Before scoring a person's shirt for colour, we first identify and exclude bright green grass pixels. Grass is high-saturation, high-value green; kit fabric green is duller. This prevents the pitch from fooling the detector. |
| Dual-Colour Scoring | Instead of just asking "is this person green?", we score for the ref's unique colour combination (e.g. green AND black). Only the ref has both. |
| Isolation Bonus | A small score boost for people who are spatially alone. Refs tend to stand apart from player clusters, so candidates far from their nearest neighbour get a bonus. |
| Upper Body Crop | We only examine the top 35% of each bounding box (the shirt region) and trim 10% off the sides to minimise background grass leaking into the colour analysis. |
| Colour Ratio | The fraction of non-grass pixels in a shirt crop that match the target colour. Higher = stronger match. |

## Video & Visualisation

Tools and concepts for processing and annotating video.

| Term | What It Means |
|------|---------------|
| OpenCV | Open Source Computer Vision library — the Swiss army knife of image/video processing. Used for reading frames, HSV conversion, colour masking, and writing annotated output video. |
| Supervision | A Python library by Roboflow for annotating detection results on video. Adds custom label styling and visual annotations that YOLO can't do natively. |
| Heatmap | A 2D visualisation showing where the ref spent most time on the pitch. Hot spots = high frequency positions. Generated from accumulated centroid positions across all frames. |
| Movement Trail | A series of dots drawn behind the referee showing their recent path across the pitch. Approximately 5 seconds of trail displayed with fading opacity. |
| Centroid | The centre point of a bounding box (midpoint of x and y coordinates). Tracked frame-by-frame to build movement trails and heatmaps. |
| FPS | Frames Per Second. Veo footage typically runs at 29fps, meaning 29 individual images per second of video. |

## AI & Decision Classification

The enrichment layer that explains what the referee is doing.

| Term | What It Means |
|------|---------------|
| Gemini 2.5 Flash | Google's multimodal AI model. We send it video segments and ask it to classify referee decisions and explain them in plain English. Free tier: 250 requests/day. |
| Multimodal | An AI that can process multiple types of input — text, images, video, audio. Gemini is multimodal because it can watch a video clip and describe what's happening. |
| Decision Classification | Categorising each referee action into types: penalty, free kick, try, scrum, knock on, advantage, yellow/red card, etc. |
| Signal Reference Images | Official World Rugby referee signal diagrams included alongside video when calling Gemini. Maps visual gestures to decision types so Gemini can match what it sees against known signal patterns. |
| Enrichment Layer | Gemini adds context and explanation on top of the core CV tracking. The project works without it (you still get tracking + heatmap), but Gemini makes it understandable to non-experts. |

## Tools & Infrastructure

| Term | What It Means |
|------|---------------|
| uv | A fast Python package manager (replacement for pip). Keeps dependencies in a project-specific virtual environment. Created by Astral. |
| Veo | A camera system used by grassroots rugby/football clubs. Records from an elevated wide-angle position — exactly what we need because the ref stays in frame. |
| Wide-Angle Footage | Video shot from a fixed, elevated camera that captures the whole pitch. Essential for ref tracking because the ref is always visible. Broadcast highlights don't work — the camera follows the ball and the ref is off-screen most of the time. |
| Pre-trained Model | A model that has already learned from a large dataset (like COCO). Used directly without additional training — it already knows how to detect people. |
| Inference | Running a trained model on new data to get predictions. When YOLOv8 processes a frame and outputs bounding boxes, that's inference. |
| BoxMOT | A modular multi-object tracking library supporting BoTSORT, DeepOCSORT, StrongSORT, and others. Integrates with YOLOv8 and supports multiple ReID backends. |

## The RefTracker Pipeline

| Step | What Happens |
|------|-------------|
| 1. Track referee (local) | YOLOv8 detects people → grass-masked HSV colour filter identifies ref → BoTSORT/CLIP-ReID maintains identity → Supervision draws annotations |
| 2. Classify decisions (API) | Gemini 2.5 Flash watches video segments, classifies ref decisions, explains in plain English |
| 3. Merge & annotate | Overlay decision labels + explanations onto tracked video with banners |
| 4. Output | Annotated .mp4 video + movement heatmap .png + tracking .json data |

Total cost: $0. YOLOv8, OpenCV, Supervision, and BoxMOT are free and open source. Gemini 2.5 Flash free tier provides 250 requests/day.
