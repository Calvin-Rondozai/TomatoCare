from typing import Any, Dict, List


DISEASES: List[Dict[str, Any]] = [
    {
        "diseaseName": "Tomato Healthy",
        "scientificName": "Solanum lycopersicum",
        "severity": "Low",
        "summary": "The leaf appears healthy with no visible disease symptoms.",
        "symptoms": [
            "Uniform green color",
            "No spots or lesions",
            "No yellowing or wilting",
        ],
        "recommendation": (
            "Maintain good watering and fertilization practices. "
            "Continue monitoring plants regularly for early signs of disease."
        ),
    },
    {
        "diseaseName": "Tomato Early Blight",
        "scientificName": "Alternaria solani",
        "severity": "Moderate",
        "summary": (
            "Early blight is a common fungal disease of tomato that causes dark "
            "lesions with concentric rings, usually starting on older leaves."
        ),
        "symptoms": [
            "Dark brown spots with target-like concentric rings",
            "Yellowing tissue around lesions",
            "Premature leaf drop from the bottom of the plant upward",
        ],
        "recommendation": (
            "Remove and destroy heavily infected leaves. Avoid overhead watering. "
            "Apply an appropriate fungicide following local guidelines and rotate "
            "crops to reduce inoculum in the soil."
        ),
    },
    {
        "diseaseName": "Tomato Leaf Spot",
        "scientificName": "Septoria lycopersici",
        "severity": "Moderate",
        "summary": (
            "Septoria leaf spot is a fungal disease that produces many small, "
            "circular spots with dark margins and light centers on leaves."
        ),
        "symptoms": [
            "Numerous small circular spots with gray centers",
            "Dark brown or purple borders around spots",
            "Progressive yellowing and dropping of lower leaves",
        ],
        "recommendation": (
            "Prune and remove affected foliage, improve air circulation, avoid "
            "wetting leaves, and apply fungicides where recommended."
        ),
    },
    {
        "diseaseName": "Tomato Late Blight",
        "scientificName": "Phytophthora infestans",
        "severity": "Severe",
        "summary": (
            "Late blight is a destructive disease that can rapidly destroy tomato "
            "foliage and fruit under cool, wet conditions."
        ),
        "symptoms": [
            "Water-soaked lesions that turn dark and enlarge quickly",
            "White fungal growth on undersides of leaves in humid conditions",
            "Brown, firm lesions on stems and fruit",
        ],
        "recommendation": (
            "Immediately remove and destroy infected plants. Avoid overhead "
            "irrigation, increase spacing, and apply recommended fungicides. "
            "Do not save seed or tubers from infected plants."
        ),
    },
]


def get_disease_metadata(index: int) -> Dict[str, Any]:
    """
    Return metadata for the given class index.

    If the index is outside the configured range, we still return a safe,
    generic response so the mobile app can render a valid Results screen.
    """
    if 0 <= index < len(DISEASES):
        return DISEASES[index]

    return {
        "diseaseName": f"Class {index}",
        "scientificName": "",
        "severity": "Unknown",
        "summary": (
            "No detailed description is configured for this class index. "
            "Please consult a local expert or agronomist for diagnosis."
        ),
        "symptoms": [],
        "recommendation": (
            "Capture additional images and seek advice from an agricultural "
            "extension service or plant pathologist."
        ),
    }
