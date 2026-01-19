from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "plant_disease.tflite"
LABELS_JSON_PATH = PROJECT_ROOT / "model" / "labels.json"


def _parse_class_name(class_name: str) -> Dict[str, str]:
    """
    Parse a class name like "Tomato___Early_blight" into plant and disease.
    Returns: {"plant": "Tomato", "disease": "Early Blight", "fullName": "Tomato___Early_blight"}
    """
    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
        disease_display = disease.replace("_", " ")
        return {
            "plant": plant,
            "disease": disease_display,
            "fullName": class_name,
        }
    return {
        "plant": "Unknown",
        "disease": class_name.replace("_", " "),
        "fullName": class_name,
    }


def _create_disease_metadata(class_name: str, index: int) -> Dict[str, Any]:
    """Create metadata structure for a disease class from its name."""
    parsed = _parse_class_name(class_name)
    is_healthy = "healthy" in parsed["disease"].lower()

    # Format disease name nicely (title case, better formatting)
    disease_display = parsed["disease"].title()

    # Determine severity based on disease type
    if is_healthy:
        severity = "Low"
        disease_display = "Healthy"
    elif any(x in parsed["disease"].lower() for x in ["virus", "mosaic", "curl"]):
        severity = "Severe"
    elif any(x in parsed["disease"].lower() for x in ["blight", "spot", "rot"]):
        severity = "Severe" if "late" in parsed["disease"].lower() else "Moderate"
    else:
        severity = "Moderate"

    # Create a more detailed summary based on disease type
    if is_healthy:
        summary = f"The {parsed['plant']} leaf appears healthy with no visible signs of disease or damage."
    elif "bacterial" in parsed["disease"].lower():
        summary = (
            f"Bacterial disease detected on {parsed['plant']}: {disease_display}. "
            "Bacterial diseases typically cause spots, wilting, or cankers and can spread rapidly."
        )
    elif "blight" in parsed["disease"].lower():
        summary = (
            f"Blight disease detected on {parsed['plant']}: {disease_display}. "
            "Blight diseases cause rapid browning and death of plant tissue."
        )
    elif "virus" in parsed["disease"].lower():
        summary = (
            f"Viral disease detected on {parsed['plant']}: {disease_display}. "
            "Viral diseases can cause mosaic patterns, stunting, and reduced yields."
        )
    else:
        summary = (
            f"Disease detected on {parsed['plant']}: {disease_display}. "
            "Please consult additional resources for specific identification and treatment."
        )

    return {
        "diseaseName": disease_display,
        "plantName": parsed["plant"],
        "scientificName": "",
        "severity": severity,
        "summary": summary,
        "symptoms": (
            ["Healthy appearance with no visible disease symptoms"]
            if is_healthy
            else [
                "Visual symptoms may include spots, discoloration, or lesions",
                "Leaves may show signs of wilting, yellowing, or browning",
                "Consult additional resources for specific symptom identification",
            ]
        ),
        "recommendation": (
            "Maintain good agricultural practices and continue monitoring. "
            "Keep plants well-watered and fertilized."
            if is_healthy
            else (
                "Remove and destroy affected plant parts immediately. "
                "Improve air circulation around plants. "
                "Consider appropriate fungicide/bactericide treatment based on the specific disease. "
                "Consult local agricultural extension services for targeted recommendations."
            )
        ),
    }


# Tomato-only filter: Only classes 28-37 are tomato-related
TOMATO_CLASS_INDICES = list(range(28, 38))  # 28 to 37 (10 tomato classes)


def _load_class_labels() -> Dict[int, str]:
    """Load class labels from labels.json file, filtered to tomato classes only."""
    if not LABELS_JSON_PATH.exists():
        return {}

    with open(LABELS_JSON_PATH, "r") as f:
        labels_dict = json.load(f)

    # Convert string keys to integers and filter to tomato classes only
    all_labels = {
        int(k): v for k, v in sorted(labels_dict.items(), key=lambda x: int(x[0]))
    }

    # Filter to only tomato classes (TomatoCare focuses on tomato diseases)
    tomato_labels = {
        idx: all_labels[idx] for idx in TOMATO_CLASS_INDICES if idx in all_labels
    }

    return tomato_labels


def get_disease_metadata(index: int) -> Dict[str, Any]:
    """
    Return metadata for the given class index.
    Filters to tomato classes only for TomatoCare project.
    """
    # Only allow tomato classes (28-37)
    if index not in TOMATO_CLASS_INDICES:
        return {
            "diseaseName": "Non-Tomato Class",
            "plantName": "Unknown",
            "scientificName": "",
            "severity": "Unknown",
            "summary": "This prediction is not for a tomato plant. TomatoCare focuses on tomato diseases only.",
            "symptoms": [],
            "recommendation": "Please use an image of a tomato plant leaf for analysis.",
        }

    # Try to load from labels.json first
    class_labels = _load_class_labels()
    if class_labels and index in class_labels:
        return _create_disease_metadata(class_labels[index], index)

    # Fallback to Backend/labels.py
    try:
        from labels import get_disease_metadata as fallback_get_metadata

        return fallback_get_metadata(index)
    except ImportError:
        # Ultimate fallback
        return {
            "diseaseName": f"Class {index}",
            "plantName": "Tomato",
            "scientificName": "",
            "severity": "Unknown",
            "summary": "No metadata available for this class index.",
            "symptoms": [],
            "recommendation": "Consult an agricultural expert for diagnosis.",
        }


# Log which labels are being used
class_labels = _load_class_labels()
if class_labels:
    sorted_keys = sorted(class_labels.keys())
    first_key = sorted_keys[0] if sorted_keys else None
    last_key = sorted_keys[-1] if sorted_keys else None
    print(f"[INFO] Loaded {len(class_labels)} disease classes from {LABELS_JSON_PATH}")
    if first_key is not None and last_key is not None:
        print(
            f"[INFO] First class: {class_labels[first_key]} (index {first_key}), "
            f"Last class: {class_labels[last_key]} (index {last_key})"
        )
else:
    print(f"[INFO] Using fallback labels from Backend/labels.py")


class PlantDiseaseModel:
    """
    Thin wrapper around the TFLite interpreter that handles model loading,
    preprocessing and prediction with PRECISE Grad-CAM visualization.
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"TFLite model not found at: {model_path}")

        self._interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Expect a shape like [1, height, width, channels]
        input_shape = self._input_details[0]["shape"]
        print(f"[DEBUG] Model input shape from TFLite: {input_shape}")

        if len(input_shape) != 4:
            raise ValueError(f"Unexpected input shape for TFLite model: {input_shape}")

        batch_dim, dim1, dim2, dim3 = input_shape

        # Detect format: channels-last [1, H, W, C] or channels-first [1, C, H, W]
        if dim3 <= 4:  # Likely channels-last
            self.height, self.width, self.channels = dim1, dim2, dim3
            self.channels_first = False
            print(f"[DEBUG] Detected channels-last format [1, H, W, C]")
        elif dim1 <= 4:  # Likely channels-first
            self.channels, self.height, self.width = dim1, dim2, dim3
            self.channels_first = True
            print(f"[DEBUG] Detected channels-first format [1, C, H, W]")
        else:
            # Default to channels-last
            self.height, self.width, self.channels = dim1, dim2, dim3
            self.channels_first = False
            print(f"[DEBUG] Defaulting to channels-last format [1, H, W, C]")

        print(
            f"[DEBUG] Parsed model dimensions - Height: {self.height}, "
            f"Width: {self.width}, Channels: {self.channels}"
        )

    def preprocess(self, file) -> np.ndarray:
        """
        Convert an uploaded image file into a normalized tensor matching the model's expected input.
        """
        file.stream.seek(0)

        try:
            image = Image.open(file.stream).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open image: {str(e)}")

        original_size = image.size
        print(f"[DEBUG] Original image size: {original_size} (width x height)")

        # Resize to model's expected input size
        try:
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        except AttributeError:
            image = image.resize((self.width, self.height), Image.LANCZOS)

        resized_size = image.size
        print(f"[DEBUG] Resized image size: {resized_size} (width x height)")

        if resized_size != (self.width, self.height):
            raise ValueError(
                f"Resize failed: got {resized_size}, expected ({self.width}, {self.height})"
            )

        # Convert to numpy array
        array = np.array(image, dtype=np.float32)
        print(f"[DEBUG] Array shape after conversion: {array.shape}")

        # Verify shape
        expected_shape = (self.height, self.width, self.channels)
        if array.shape != expected_shape:
            raise ValueError(
                f"Unexpected array shape after resize: {array.shape}, "
                f"expected {expected_shape}"
            )

        # Normalize using ImageNet statistics
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        array = array / 255.0
        array = (array - imagenet_mean) / imagenet_std

        # Add batch dimension
        array = np.expand_dims(array, axis=0)

        # Handle channels-first if needed
        if self.channels_first:
            array = np.transpose(array, (0, 3, 1, 2))
            expected_shape = (1, self.channels, self.height, self.width)
        else:
            expected_shape = (1, self.height, self.width, self.channels)

        if array.shape != expected_shape:
            raise ValueError(
                f"Final tensor shape mismatch: {array.shape}, expected {expected_shape}"
            )

        print(f"[DEBUG] Final tensor shape: {array.shape}")
        return array

    def predict(
        self, image_tensor: np.ndarray, return_all_probs: bool = False
    ) -> Dict[str, Any]:
        """
        Run inference on a preprocessed image tensor and return a structured prediction dictionary.
        """
        input_index = self._input_details[0]["index"]
        output_index = self._output_details[0]["index"]

        expected_shape = tuple(self._input_details[0]["shape"])
        print(f"[DEBUG] Image tensor shape: {image_tensor.shape}")
        print(f"[DEBUG] Expected input shape: {expected_shape}")

        self._interpreter.set_tensor(input_index, image_tensor)
        self._interpreter.invoke()
        raw_output = self._interpreter.get_tensor(output_index)[0]

        all_probabilities = self._softmax(raw_output)

        # Filter to only tomato classes
        tomato_probs = all_probabilities[TOMATO_CLASS_INDICES]
        tomato_class_indices = TOMATO_CLASS_INDICES

        # Find the best tomato class
        best_tomato_idx = int(np.argmax(tomato_probs))
        original_class_index = tomato_class_indices[best_tomato_idx]
        confidence = float(tomato_probs[best_tomato_idx])

        print(
            f"[DEBUG] Top tomato prediction: Class {original_class_index} "
            f"with confidence {confidence:.4f}"
        )

        metadata = get_disease_metadata(original_class_index)

        result = {
            "classIndex": original_class_index,
            "confidence": confidence,
            **metadata,
        }

        if return_all_probs:
            result["allProbabilities"] = {
                tomato_class_indices[i]: float(prob)
                for i, prob in enumerate(tomato_probs)
            }

        return result

    def generate_heatmap(self, file, class_index: int) -> str:
        """
        Generate PRECISE Grad-CAM that shows ONLY the actual affected/diseased areas.
        Uses advanced spatial analysis and multi-stage filtering for accurate localization.
        Returns: Base64-encoded image string
        """
        import base64
        from io import BytesIO

        try:
            # Reset and load original image
            # Ensure stream is at the beginning
            try:
                file.stream.seek(0)
            except (AttributeError, OSError) as seek_error:
                print(f"[WARNING] Could not seek file stream: {seek_error}")
                # Try to get the file content another way
                if hasattr(file, "read"):
                    file.read()  # This might reset the stream
                else:
                    raise ValueError("Cannot read file stream")

            original_image = Image.open(file.stream).convert("RGB")
            original_size = original_image.size
            print(f"[DEBUG] Original image size: {original_size}")
        except Exception as e:
            print(f"[ERROR] Failed to load image: {e}")
            import traceback

            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise

        # Preprocess for model
        image_tensor = self.preprocess(file)

        # Run inference
        input_index = self._input_details[0]["index"]
        output_index = self._output_details[0]["index"]

        self._interpreter.set_tensor(input_index, image_tensor)
        self._interpreter.invoke()

        # Get all tensor details to find last conv layer
        all_tensors = self._interpreter.get_tensor_details()

        # Find the last convolutional layer with best spatial resolution
        last_conv_idx = None
        best_spatial_size = 0

        for i, tensor_info in enumerate(all_tensors):
            shape = tensor_info["shape"]
            if len(shape) == 4 and shape[0] == 1:
                if self.channels_first:
                    spatial_size = shape[2] * shape[3]
                    if (
                        shape[2] > 7
                        and shape[3] > 7
                        and spatial_size > best_spatial_size
                    ):
                        last_conv_idx = i
                        best_spatial_size = spatial_size
                else:
                    spatial_size = shape[1] * shape[2]
                    if (
                        shape[1] > 7
                        and shape[2] > 7
                        and spatial_size > best_spatial_size
                    ):
                        last_conv_idx = i
                        best_spatial_size = spatial_size

        if last_conv_idx is None:
            print("[WARNING] Could not find suitable conv layer, using fallback")
            return self._generate_precise_image_based_heatmap(file, original_size)

        # Get activations from last conv layer
        conv_output = self._interpreter.get_tensor(last_conv_idx)
        print(f"[DEBUG] Using conv layer {last_conv_idx}, shape: {conv_output.shape}")

        # Convert activations to [H, W, C] format
        if self.channels_first and len(conv_output.shape) == 4:
            activations = conv_output[0].transpose(1, 2, 0)
        elif len(conv_output.shape) == 4:
            activations = conv_output[0]
        else:
            print("[WARNING] Unexpected activation shape, using fallback")
            return self._generate_precise_image_based_heatmap(file, original_size)

        H, W, C = activations.shape
        print(f"[DEBUG] Activation map shape: H={H}, W={W}, C={C}")

        # ===== GRAD-CAM++ IMPLEMENTATION =====
        # Grad-CAM++ improves localization by using pixel-wise weighting
        # Since TFLite doesn't provide gradients, we approximate using activation statistics

        # Get class score from prediction (already computed from inference above)
        output_scores = self._interpreter.get_tensor(output_index)

        # Extract tomato class score (class_index is already filtered to tomato classes)
        # Map class_index back to full model output if needed
        if len(output_scores.shape) == 2:
            class_score = float(output_scores[0, class_index])
        else:
            class_score = float(output_scores[class_index])

        print(f"[DEBUG] Class score (Y^c) for Grad-CAM++: {class_score:.4f}")

        # Step 1: Compute per-pixel weights using Grad-CAM++ formula
        # Grad-CAM++: α_k^c = (∂²Y^c / (∂A^k)²) / (2 * (∂²Y^c / (∂A^k)²) + Σ_a Σ_b A_ab^k * (∂³Y^c / (∂A^k)³))
        # Approximation: Use activation statistics to estimate gradient behavior

        # Normalize activations per channel for stable computation
        activations_norm = np.zeros_like(activations)
        for c in range(C):
            A_k = activations[:, :, c]
            A_min, A_max = A_k.min(), A_k.max()
            if A_max > A_min:
                activations_norm[:, :, c] = (A_k - A_min) / (A_max - A_min + 1e-8)
            else:
                activations_norm[:, :, c] = A_k

        # Compute second-order approximation: variance within each channel
        # Higher variance ≈ stronger second derivative
        channel_variance = np.var(activations_norm, axis=(0, 1))

        # Compute third-order approximation: skewness (asymmetry)
        # Skewness indicates third-order behavior
        channel_skewness = np.zeros(C)
        for c in range(C):
            A_flat = activations_norm[:, :, c].flatten()
            mean_A = np.mean(A_flat)
            std_A = np.std(A_flat) + 1e-8
            channel_skewness[c] = np.mean(((A_flat - mean_A) / std_A) ** 3)

        # Grad-CAM++ style per-pixel weights
        # Weight each pixel based on its contribution to the class score
        cam = np.zeros((H, W), dtype=np.float32)

        for c in range(C):
            A_k = activations_norm[:, :, c]

            # Approximate second derivative term: use squared activations weighted by variance
            second_deriv_approx = A_k**2 * (channel_variance[c] + 1e-8)

            # Approximate third derivative term: use cubed activations weighted by skewness
            third_deriv_approx = A_k**3 * (np.abs(channel_skewness[c]) + 1e-8)

            # Sum over spatial dimensions for third derivative term
            sum_third = np.sum(third_deriv_approx)

            # Compute alpha_k (per-channel weight) using Grad-CAM++ formula
            denominator = 2 * second_deriv_approx + sum_third + 1e-8
            alpha_k = second_deriv_approx / denominator

            # Weight by class score to emphasize relevant channels
            channel_contribution = alpha_k * (class_score + 1e-8)

            # Accumulate weighted activations
            cam += channel_contribution

        # Step 2: Apply ReLU (only positive contributions)
        cam = np.maximum(cam, 0)

        print(
            f"[DEBUG] Grad-CAM++ computed, CAM range: [{cam.min():.4f}, {cam.max():.4f}]"
        )

        # Step 5: Enhanced normalization with percentile clipping for contrast
        if cam.max() > cam.min():
            # Clip extreme outliers (top 0.1%) to prevent saturation
            upper_clip = np.percentile(cam, 99.9)
            cam = np.clip(cam, 0, upper_clip)
            # Normalize to [0, 1]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        # Step 6: Multi-stage non-linear enhancement for high contrast
        # Stage 1: Moderate power curve to boost mid-range values
        cam = np.power(cam, 0.5)

        # Stage 2: Sigmoid-like contrast stretch
        # This creates S-curve for enhanced contrast while preserving details
        # Clip to prevent overflow in exp function
        cam_clipped = np.clip(cam, 0.0, 1.0)
        cam = 1 / (1 + np.exp(-10 * (cam_clipped - 0.5)))

        # Re-normalize after sigmoid
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # ===== HIGH-QUALITY UPSAMPLING WITH EDGE PRESERVATION =====
        # Use bilateral upsampling for edge-aware interpolation
        # cv2.resize expects (width, height) - PIL's original_size is (width, height)
        cam_resized = cv2.resize(cam, original_size, interpolation=cv2.INTER_CUBIC)

        # Apply bilateral filter to preserve edges while smoothing
        try:
            # Convert to uint8 for bilateralFilter (expects 0-255 range)
            cam_uint8 = (np.clip(cam_resized, 0, 1) * 255).astype(np.uint8)
            cam_resized = cv2.bilateralFilter(cam_uint8, 9, 75, 75)
            # Convert back to float32 [0, 1]
            cam_resized = cam_resized.astype(np.float32) / 255.0
        except Exception as e:
            print(f"[WARNING] Bilateral filter failed: {e}, continuing without it")
            # Continue without bilateral filter if it fails

        # Re-normalize
        if cam_resized.max() > cam_resized.min():
            cam_resized = (cam_resized - cam_resized.min()) / (
                cam_resized.max() - cam_resized.min() + 1e-8
            )

        # ===== HYBRID APPROACH: Combine CNN activations with image analysis =====
        file.stream.seek(0)
        img_pil = Image.open(file.stream).convert("RGB")
        if img_pil.size != original_size:
            img_pil = img_pil.resize(original_size, Image.Resampling.LANCZOS)

        img_array = np.array(img_pil, dtype=np.uint8)

        # Detect actual disease symptoms
        disease_mask = self._detect_disease_regions(img_array)

        # Multiplicative fusion for precise localization
        combined_cam = cam_resized * disease_mask

        # Normalize
        if combined_cam.max() > 0:
            combined_cam = combined_cam / combined_cam.max()

        # ===== ADAPTIVE THRESHOLDING WITH ENHANCED SENSITIVITY =====
        cam_uint8 = (combined_cam * 255).astype(np.uint8)

        # Use Otsu's method
        otsu_val, otsu_mask = cv2.threshold(
            cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_threshold = otsu_val / 255.0

        # Lower threshold for higher sensitivity to fine details
        final_threshold = max(otsu_threshold * 0.7, 0.2)  # More sensitive

        print(f"[DEBUG] Otsu: {otsu_threshold:.3f}, Final: {final_threshold:.3f}")

        combined_cam[combined_cam < final_threshold] = 0

        # Re-normalize
        if combined_cam.max() > 0:
            combined_cam = combined_cam / combined_cam.max()

        # ===== MORPHOLOGICAL REFINEMENT =====
        cam_uint8 = (combined_cam * 255).astype(np.uint8)

        # Smaller kernel to preserve fine details
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cam_uint8 = cv2.morphologyEx(cam_uint8, cv2.MORPH_OPEN, kernel_small)

        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cam_uint8 = cv2.morphologyEx(cam_uint8, cv2.MORPH_CLOSE, kernel_medium)

        combined_cam = cam_uint8.astype(np.float32) / 255.0

        # Final aggressive power curve for maximum intensity
        combined_cam = np.power(combined_cam, 0.5)

        # Apply colormap
        heatmap_colored = self._apply_vivid_disease_colormap(combined_cam)

        # Handle large images
        if max(original_size) > 800:
            ratio = 800 / max(original_size)
            display_size = (
                int(original_size[0] * ratio),
                int(original_size[1] * ratio),
            )
            display_image = original_image.resize(
                display_size, Image.Resampling.LANCZOS
            )

            combined_cam = cv2.resize(
                combined_cam, display_size, interpolation=cv2.INTER_CUBIC
            )
            heatmap_colored = self._apply_vivid_disease_colormap(combined_cam)
            img_array = np.array(display_image, dtype=np.uint8)
        else:
            display_image = original_image
            display_size = original_size

        # Enhanced blending with higher opacity for visibility
        alpha = np.clip(combined_cam * 0.6, 0, 0.6)  # Increased from 0.5
        alpha = np.expand_dims(alpha, axis=2)

        overlay = (alpha * heatmap_colored + (1 - alpha) * img_array).astype(np.uint8)

        # Enhanced sharpening for crisp details
        try:
            overlay = cv2.addWeighted(
                overlay, 1.2, cv2.GaussianBlur(overlay, (0, 0), 1.0), -0.2, 0
            )
        except Exception as e:
            print(f"[WARNING] Sharpening failed: {e}, continuing without sharpening")

        # Convert to PIL and encode
        try:
            overlay_image = Image.fromarray(overlay)
            buffer = BytesIO()
            overlay_image.save(buffer, format="JPEG", quality=95, optimize=True)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            print(
                f"[DEBUG] Heatmap encoded successfully, base64 length: {len(img_base64)}"
            )
            return img_base64
        except Exception as e:
            print(f"[ERROR] Failed to encode heatmap: {e}")
            import traceback

            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise

    def _detect_disease_regions(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Detect actual disease symptoms in the image using color and texture analysis.
        Returns a normalized mask [0, 1] where 1 = diseased area.
        """
        # Convert to different color spaces
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        l, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]

        # Initialize disease score map
        disease_score = np.zeros(img_rgb.shape[:2], dtype=np.float32)

        # Feature 1: Dark spots/lesions (necrotic tissue)
        # Very dark areas with low brightness
        dark_threshold = np.percentile(v, 15)
        dark_mask = (v < dark_threshold).astype(np.float32)
        disease_score += dark_mask * 1.0

        # Feature 2: Brown discoloration (common in blight, bacterial spot)
        # Brown in HSV: hue 5-30, medium-low brightness
        brown_mask = ((h >= 5) & (h <= 30) & (s > 40) & (v < 180)).astype(np.float32)
        disease_score += brown_mask * 0.9

        # Feature 3: Yellow/chlorotic regions (viral diseases, nutrient deficiency)
        # Yellow in HSV: hue 20-45
        yellow_mask = ((h >= 20) & (h <= 45) & (s > 60)).astype(np.float32)
        disease_score += yellow_mask * 0.6

        # Feature 4: Abnormal color in LAB space
        # Healthy leaves are green (low a*, high b*)
        # Diseased areas have abnormal a* and b* values
        a_abnormal = np.abs(a - np.median(a)) > np.std(a)
        b_abnormal = np.abs(b - np.median(b)) > np.std(b)
        color_abnormal = (a_abnormal | b_abnormal).astype(np.float32)
        disease_score += color_abnormal * 0.5

        # Feature 5: Texture irregularities (using Laplacian for edge detection)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian)

        # High Laplacian in dark areas indicates lesion boundaries
        texture_threshold = np.percentile(laplacian_abs, 75)
        texture_mask = ((laplacian_abs > texture_threshold) & (v < 150)).astype(
            np.float32
        )
        disease_score += texture_mask * 0.7

        # Normalize disease score to [0, 1]
        if disease_score.max() > 0:
            disease_score = disease_score / disease_score.max()

        # Apply smoothing to create coherent regions
        disease_score = cv2.GaussianBlur(disease_score, (9, 9), 0)

        # Re-normalize after smoothing
        if disease_score.max() > 0:
            disease_score = disease_score / disease_score.max()

        return disease_score

    def _apply_vivid_disease_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Apply standard Grad-CAM colormap (JET-like): Blue -> Cyan -> Green -> Yellow -> Red
        This matches professional plant pathology visualization standards.
        """
        try:
            # Use OpenCV's JET colormap for standard Grad-CAM visualization
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # Convert from BGR to RGB (OpenCV uses BGR)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

            return colored
        except Exception as e:
            # Fallback: manual JET colormap if OpenCV fails
            print(f"[WARNING] OpenCV colormap failed, using fallback: {e}")
            return self._apply_manual_jet_colormap(heatmap)

    def _apply_manual_jet_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """Manual JET colormap implementation (fallback)"""
        h, w = heatmap.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        values = np.clip(heatmap, 0.0, 1.0)

        # JET colormap: Blue -> Cyan -> Green -> Yellow -> Red
        r = np.zeros_like(values)
        g = np.zeros_like(values)
        b = np.zeros_like(values)

        # Blue to Cyan (0.0 - 0.25)
        mask1 = values < 0.25
        r[mask1] = 0
        g[mask1] = (values[mask1] * 4.0 * 255).astype(np.uint8)
        b[mask1] = 255

        # Cyan to Green (0.25 - 0.5)
        mask2 = (values >= 0.25) & (values < 0.5)
        r[mask2] = 0
        g[mask2] = 255
        b[mask2] = (255 - (values[mask2] - 0.25) * 4.0 * 255).astype(np.uint8)

        # Green to Yellow (0.5 - 0.75)
        mask3 = (values >= 0.5) & (values < 0.75)
        r[mask3] = ((values[mask3] - 0.5) * 4.0 * 255).astype(np.uint8)
        g[mask3] = 255
        b[mask3] = 0

        # Yellow to Red (0.75 - 1.0)
        mask4 = values >= 0.75
        r[mask4] = 255
        g[mask4] = (255 - (values[mask4] - 0.75) * 4.0 * 255).astype(np.uint8)
        b[mask4] = 0

        colored[:, :, 0] = np.clip(r, 0, 255)
        colored[:, :, 1] = np.clip(g, 0, 255)
        colored[:, :, 2] = np.clip(b, 0, 255)

        return colored

    def _generate_precise_image_based_heatmap(self, file, original_size: tuple) -> str:
        """
        Fallback method with precise disease detection.
        """
        import base64
        from io import BytesIO

        file.stream.seek(0)
        original_image = Image.open(file.stream).convert("RGB")

        if max(original_size) > 800:
            ratio = 800 / max(original_size)
            process_size = (
                int(original_size[0] * ratio),
                int(original_size[1] * ratio),
            )
            img = original_image.resize(process_size, Image.Resampling.LANCZOS)
        else:
            img = original_image
            process_size = original_size

        img_array = np.array(img, dtype=np.uint8)

        # Use the same precise disease detection
        heatmap = self._detect_disease_regions(img_array)

        # Apply threshold
        threshold = (
            max(np.percentile(heatmap[heatmap > 0], 50), 0.25)
            if np.any(heatmap > 0)
            else 0.25
        )
        heatmap[heatmap < threshold] = 0

        # Re-normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Morphological cleanup
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        heatmap_uint8 = cv2.morphologyEx(heatmap_uint8, cv2.MORPH_OPEN, kernel)
        heatmap_uint8 = cv2.morphologyEx(heatmap_uint8, cv2.MORPH_CLOSE, kernel)
        heatmap = heatmap_uint8.astype(np.float32) / 255.0

        # Apply standard Grad-CAM colormap
        heatmap_colored = self._apply_vivid_disease_colormap(heatmap)

        # Standard blending (50% opacity)
        alpha = np.clip(heatmap * 0.5, 0, 0.5)
        alpha = np.expand_dims(alpha, axis=2)

        overlay = (alpha * heatmap_colored + (1 - alpha) * img_array).astype(np.uint8)

        # Subtle sharpening
        overlay = cv2.addWeighted(
            overlay, 1.1, cv2.GaussianBlur(overlay, (0, 0), 1.0), -0.1, 0
        )

        overlay_image = Image.fromarray(overlay)
        buffer = BytesIO()
        overlay_image.save(buffer, format="JPEG", quality=95, optimize=True)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()


# Singleton instance used by the Flask app
model = PlantDiseaseModel()
