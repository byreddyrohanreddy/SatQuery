# SatQuery AI — Satellite Imagery Query System

SatQuery AI is an agentic vision-language assistant and controller backend for satellite imagery analysis. It allows users to ask plain-language questions about single or bi-temporal satellite scenes (Optical, SAR, and GeoTIFF), automatically routes queries to specialized computer vision and deep-learning handlers via LLM function calling (Google Gemini or Anthropic Claude) with a deterministic fallback, and renders interactive visual bounding boxes and complete execution traces in an integrated web frontend.

---

## Features

- **Agentic Task Routing**: Uses Google Gemini (`gemini-flash-lite-latest`, `gemini-3.6-flash`) or Anthropic Claude tool-calling to extract parameters and select specialist models, with automatic offline keyword fallback.
- **Multimodal Support**: Works with standard optical imagery (RGB JPEG/PNG), Synthetic Aperture Radar (SAR grayscale speckle), and 16-bit GeoTIFF (`.tif`/`.tiff`) files.
- **Specialized Downstream Handlers**:
  - **Visual Question Answering & Captioning**: Salesforce BLIP deep vision-language models.
  - **Spatial Grounding**: Computer vision color-space (HSV) and contour detection to pinpoint and box objects (water bodies, vegetation, urban features).
  - **Bi-Temporal Change Analysis**: Difference matrix computation, morphological filtering, and spatial percentage calculations.
  - **Optical + SAR Fusion**: Cross-modality synthesis combining optical spectral variance and SAR radar backscatter.
- **Trace Transparency**: Visible-by-default audit trail displaying the exact routing path, models called, input validator outputs, and confidence basis.
- **Modern Web Interface**: Clean single-page application with drag-and-drop uploads, instant previews, animated agentic progress steps, and dynamic percentage-scaled visual overlays.

---

## Quick Start

### 1. Clone the Repository & Install Dependencies

```bash
git clone https://github.com/byreddyrohanreddy/SatQuery.git
cd SatQuery

# Recommended: create or activate your Python 3.10+ or Conda environment
conda activate ml   # or: python -m venv venv && source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Your API Key (Optional but Recommended)

SatQuery AI uses **Google Gemini** for intelligent query routing. Copy the template and add your free Gemini API key:

```bash
# Copy example configuration
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
> **Tip:** You can obtain a free Gemini API key at [Google AI Studio](https://aistudio.google.com/).  
> If you omit the API key, the system runs in **fully offline mode** using its deterministic keyword routing engine.

### 3. Launch the Application

Start the local server:
```bash
uvicorn satquery.api:app --reload --host 0.0.0.0 --port 8000
```

Open your browser and navigate to:
**[http://localhost:8000](http://localhost:8000)**

---

## How to Use the Web Interface

### Step 1: Upload Your Satellite Imagery
On the home page (`http://localhost:8000`), you will see two upload slots:
- **Image 1 (Primary / Optical / Before)**: Required for all queries. Drag and drop a satellite image (`.jpg`, `.png`, `.tif`, `.tiff`) or click to browse.
- **Image 2 (Secondary / SAR / After)**: Optional for single-image queries; required for **Bi-Temporal Change Detection** or **Optical + SAR Fusion**.

> **Quick Demo Buttons:** If you do not have satellite images on hand, click any of the three demo buttons at the top:
> - **Single Image Demo**: Loads optical imagery with a land-cover query.
> - **Optical + SAR Demo**: Loads co-registered optical and radar files.
> - **Change Detection Demo**: Loads before and after scenes of an evolving area.

---

### Step 2: Ask a Question in Plain Language
In the text area below the dropzones, enter your natural-language question. Examples:
- **Object Grounding:** `"Highlight the water body in this image"` or `"Where is the vegetation?"`
- **Scene Captioning / VQA:** `"Describe the terrain in this satellite scene"` or `"Are there runways visible?"`
- **Change Analysis (2 images):** `"What changed between these two acquisition dates?"`
- **Sensor Fusion (Optical + SAR):** `"Combine the optical and SAR imagery and describe what each sensor reveals."`

---

### Step 3: Run Query & Watch the Agent Work
- Click **Run Query**.
- An animated agentic status panel will display the controller pipeline in real time:
  1. *Validating input* (modality detection & dimension checking)
  2. *Routing to specialist model* (Gemini tool-calling / function declaration)
  3. *Running analysis* (specialist model inference)

---

### Step 4: Explore the Results Page
The application automatically transitions to `results.html`:

1. **Imagery & Visual Evidence**:
   - Interactive normalized bounding boxes and change region outlines dynamically overlaid on top of your imagery.
   - For change analysis, regions are mapped across both before and after frames.
2. **Answer Card**:
   - Synthesized natural language explanation from the specialist vision model.
3. **Calibrated Confidence**:
   - Visual percentage meter with an explicit **Basis** explanation detailing why that score was assigned.
4. **How This Answer Was Produced (Execution Trace)**:
   - **Task selected**: e.g., `Grounding (object localization)` or `Change analysis`.
   - **Routing path**: Colored pill indicating `LLM routed` or `Fallback routed` (with fallback reason).
   - **Model(s) called**: The exact underlying computer vision or neural network model (e.g. `Salesforce/blip-vqa-base` or `OpenCV HSV Grounding Handler`).
   - **Validator output**: Detected modalities, image count, and resolution checks.
   - **Parameters used**: Structured arguments passed from the router into the handler.

---

### Step 5: Run Another Query
Click **Run Another Query** at the bottom of the results page to return to the upload interface with freshly reset session state.

---

## Architecture Overview

```
             User Query + Satellite Image(s)
                           │
                           ▼
               ┌───────────────────────┐
               │    Input Validator    │  ← Format check, Modality detection (Optical/SAR),
               └───────────┬───────────┘    Aspect-ratio & scene geometry verification
                           │ ValidationResult
                           ▼
               ┌───────────────────────┐
               │  Agentic Task Router  │  ← PRIMARY: Google Gemini Function Calling
               └───────────┬───────────┘    FALLBACK: Deterministic Keyword Engine
                           │ task_name + extracted arguments
                           ▼
               ┌───────────────────────┐
               │    Model Registry     │  ← Dynamic dispatch to specialist handlers
               └───────────┬───────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────────────┐
        │               Specialist Handlers               │
        │  • vqa_caption        (Salesforce BLIP models)  │
        │  • grounding          (OpenCV HSV Contours)     │
        │  • change_analysis    (Bi-temporal Differencing)│
        │  • optical_sar_fusion (Dual-Modality Synthesis) │
        └──────────────────┬──────────────────────────────┘
                           │ HandlerResult
                           ▼
               ┌───────────────────────┐
               │   Confidence Scorer   │  ← Calibrated heuristic & basis formulation
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │ Execution Trace Build │  ← Full audit trail: routing, models, params
               └───────────┬───────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────────────┐
        │               Interactive Web UI                │
        │  • Answer Card                                  │
        │  • Scaled Bounding Box Overlays                 │
        │  • Confidence Gauge with Basis                  │
        │  • Transparent Audit Trail Panel                │
        └─────────────────────────────────────────────────┘
```

---

## Running CLI & Tests

### Standalone CLI Pipeline (No Server Required)
```bash
# Test grounding on a single image
python test_pipeline.py --images samples/sample_single_optical.jpg --query "Highlight the water body"

# Test change detection on two images
python test_pipeline.py --images samples/sample_before.jpg samples/sample_after.jpg --query "What changed between these dates?"
```

### Automated Test Suite
Run the 49 unit and integration tests:
```bash
pytest tests/ -v
```

---

## Interactive API Documentation

With the server running, access the interactive Swagger OpenAPI docs at:
- **Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## License

MIT License. Open source for academic, research, and commercial satellite imagery exploration.
