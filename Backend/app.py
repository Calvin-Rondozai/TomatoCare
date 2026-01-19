from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

from model_service import model


def create_app() -> Flask:
    app = Flask(__name__)

    # Allow the mobile app (running on a different host/port) to call the API.
    # Explicitly allow all methods and headers for CORS
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": "*",
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Accept"],
            }
        },
        supports_credentials=True,
    )

    @app.route("/api/health", methods=["GET"])
    def health() -> Dict[str, Any]:
        return jsonify({"status": "ok"}), 200

    @app.route("/api/analyze", methods=["POST", "OPTIONS"])
    def analyze() -> Any:
        """
        Accept an uploaded leaf image and return plant disease prediction data.

        Expected request (multipart/form-data):
          - image: file
        """
        # Handle CORS preflight
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200

        # Log request info for debugging
        print(
            f"[DEBUG] Received request - Method: {request.method}, Content-Type: {request.content_type}"
        )
        print(f"[DEBUG] Files in request: {list(request.files.keys())}")
        print(f"[DEBUG] Form data keys: {list(request.form.keys())}")
        print(f"[DEBUG] Request remote address: {request.remote_addr}")
        print(f"[DEBUG] Request headers: {dict(request.headers)}")

        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' file in request"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Handle empty body or zero-length file gracefully
        try:
            file.stream.seek(0, 2)  # Seek to end
            size = file.stream.tell()
            file.stream.seek(0)  # Reset to beginning
            if size == 0:
                return jsonify({"error": "Uploaded file is empty"}), 400
            print(f"[DEBUG] File size: {size} bytes")
        except Exception as e:
            print(f"[WARNING] Could not determine file size: {e}, continuing anyway")
            file.stream.seek(0)  # Ensure we're at the beginning

        try:
            print("[DEBUG] Starting image preprocessing...")
            image_tensor = model.preprocess(file)
            print("[DEBUG] Image preprocessed successfully")

            print("[DEBUG] Starting prediction...")
            prediction = model.predict(image_tensor, return_all_probs=True)
            print(f"[DEBUG] Prediction complete: {prediction.get('diseaseName')}")

            # Generate Grad-CAM heatmap (with timeout protection)
            try:
                print(
                    f"[DEBUG] Generating heatmap for class index: {prediction['classIndex']}"
                )
                # Reset file stream for heatmap generation
                file.stream.seek(0)

                # Generate heatmap with error handling
                heatmap_base64 = model.generate_heatmap(file, prediction["classIndex"])

                if heatmap_base64:
                    prediction["heatmapBase64"] = heatmap_base64
                    print(
                        f"[DEBUG] Heatmap generated successfully, length: {len(heatmap_base64)}"
                    )
                else:
                    print("[WARNING] Heatmap generation returned None")
                    prediction["heatmapBase64"] = None
            except Exception as heatmap_error:
                import traceback

                print(f"[ERROR] Could not generate heatmap: {heatmap_error}")
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                # Continue without heatmap - don't fail the entire request
                prediction["heatmapBase64"] = None
        except Exception as exc:  # noqa: BLE001
            # In production you would log the full traceback.
            import traceback

            error_traceback = traceback.format_exc()
            print(f"[ERROR] Failed to analyze image: {exc}")
            print(f"[ERROR] Full traceback:\n{error_traceback}")
            return (
                jsonify(
                    {
                        "error": "Failed to analyze image",
                        "details": str(exc),
                    }
                ),
                500,
            )

        # Shape the response so it matches what the ResultsScreen expects.
        response = {
            "diseaseName": prediction.get("diseaseName"),
            "scientificName": prediction.get("scientificName"),
            "confidence": prediction.get("confidence"),
            "severity": prediction.get("severity"),
            "summary": prediction.get("summary"),
            "symptoms": prediction.get("symptoms") or [],
            "recommendation": prediction.get("recommendation"),
            # Extra fields (optional but useful for debugging)
            "classIndex": prediction.get("classIndex"),
        }

        # Add heatmap if available (just the base64 string, not full data URI)
        if "heatmapBase64" in prediction and prediction["heatmapBase64"] is not None:
            try:
                # Extract just the base64 part if it's a full data URI
                heatmap_str = prediction["heatmapBase64"]
                if heatmap_str.startswith("data:image"):
                    # Extract base64 part after comma
                    heatmap_str = heatmap_str.split(",", 1)[1]
                response["heatmapBase64"] = heatmap_str
                # Log heatmap size for debugging
                print(f"[DEBUG] Heatmap base64 length: {len(heatmap_str)} characters")
            except Exception as e:
                print(f"[WARNING] Error processing heatmap: {e}")
                # Continue without heatmap

        # Set response headers to prevent timeout issues
        flask_response = jsonify(response)
        flask_response.headers["Access-Control-Allow-Origin"] = "*"
        flask_response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        flask_response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
        flask_response.headers["Content-Type"] = "application/json"

        return flask_response, 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
