"""FastAPI backend for Maharashtra Cotton, Soybean, and Onion disease detection.

Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crop_diagnosis")
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
TARGET_CROPS = ("Cotton", "Soybean", "Onion")
HELPLINE = "1800-180-1551"


class Settings(BaseSettings):
    """Runtime configuration. Environment variables override .env values."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    vision_provider: Literal["gemini", "openai"] = "gemini"
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    max_upload_bytes: int = 10 * 1024 * 1024


settings = Settings()


class Remedies(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    organic: list[str] = Field(default_factory=list)
    chemical: list[str] = Field(default_factory=list)
    preventive_measures: list[str] = Field(default_factory=list)


class MarathiTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    crop_name: Literal["कापूस", "सोयाबीन", "कांदा", "इतर पीक", "वनस्पती नाही"]
    disease_name: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=600)
    organic_remedies: list[str] = Field(default_factory=list)
    chemical_remedies: list[str] = Field(default_factory=list)
    preventive_measures: list[str] = Field(default_factory=list)


class DiagnosisResponse(BaseModel):
    """Strict model returned to every frontend client."""
    model_config = ConfigDict(extra="forbid", strict=True)
    is_target_crop: bool
    crop_name: Literal["Cotton", "Soybean", "Onion", "Other", "Not a plant"]
    disease_name: str = Field(min_length=1, max_length=200)
    confidence_score: float = Field(ge=0.0, le=1.0)
    severity_level: Literal["Mild", "Moderate", "Severe", "None"]
    affected_area_percentage: float = Field(ge=0, le=100)
    visual_symptoms: list[str] = Field(default_factory=list, max_length=20)
    remedies: Remedies
    marathi_translation: MarathiTranslation
    escalate_to_expert: bool
    helpline_number: Literal["1800-180-1551"] = HELPLINE

    @field_validator("visual_symptoms")
    @classmethod
    def reject_blank_symptoms(cls, value: list[str]) -> list[str]:
        if any(not symptom.strip() for symptom in value):
            raise ValueError("visual_symptoms cannot contain blank strings")
        return value

    @model_validator(mode="after")
    def check_target_flag(self) -> "DiagnosisResponse":
        if self.is_target_crop != (self.crop_name in TARGET_CROPS):
            raise ValueError("is_target_crop must match crop_name")
        return self


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: str
    detail: str


SYSTEM_PROMPT = """You are an agricultural plant pathologist for Maharashtra. Diagnose strictly only Cotton/कापूस, Soybean/सोयाबीन, and Onion/कांदा. Recognised diseases: Cotton—Pink Bollworm (गुलाबी बोंडअळी), Bacterial Blight/Angular Leaf Spot (जिवाणू करपा), Leaf Curl Virus (पर्णगुच्छ), Grey Mildew (दहिया रोग), Aphids/Whiteflies (मावा/पांढरी माशी). Soybean—Rust (तांबेरा), Yellow Mosaic Virus (पिवळा मोझॅक), Charcoal Rot, Anthracnose (अँथ्रॅकनोज), Stem Fly/Girdle Beetle (खोडमाशी). Onion—Purple Blotch (जांभळा करपा), Stemphylium Blight, Downy Mildew (केवडा), Thrips (फुलकिडे), Basal Rot (सड).
Return JSON ONLY, no Markdown, exactly with keys: is_target_crop (boolean), crop_name (Cotton|Soybean|Onion|Other|Not a plant), disease_name, confidence_score (0..1), severity_level (Mild|Moderate|Severe|None), affected_area_percentage (0..100), visual_symptoms (array), remedies {organic, chemical, preventive_measures}, marathi_translation {crop_name (कापूस|सोयाबीन|कांदा|इतर पीक|वनस्पती नाही), disease_name, summary, organic_remedies, chemical_remedies, preventive_measures}, escalate_to_expert (boolean), helpline_number (1800-180-1551). For Other or Not a plant set is_target_crop false, disease_name to 'Target crop not detected', severity None, affected area 0, empty remedies, and politely request a clear target-crop photograph. For blurry or uncertain photographs do not guess: low confidence and escalate_to_expert true. Chemical advice must state dosage per 15L pump or per litre; do not invent dosage when uncertain."""


def error_response(code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=code, content=ErrorResponse(error=error, detail=detail).model_dump())


async def read_and_verify_image(upload: UploadFile) -> tuple[bytes, str]:
    """Enforce content type, byte limit, and image decodability before AI inference."""
    if upload.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(415, "Only JPEG, PNG, and WebP images are supported.")
    data = await upload.read(settings.max_upload_bytes + 1)
    if not data:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "Image exceeds the 10 MB upload limit.")
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(422, "The uploaded file is corrupted or is not a valid image.") from exc
    return data, upload.content_type


def extract_json(text: str) -> dict[str, Any]:
    """Safely parse JSON, accepting only an accidental surrounding code fence."""
    content = text.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Provider JSON response must be an object")
    return value


def _sync_gemini_call(image: bytes, mime_type: str) -> str:
    """Blocking Gemini SDK call. Must run inside asyncio.to_thread, never directly
    inside an async function, since the SDK call itself is synchronous/blocking."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.google_api_key)

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            SYSTEM_PROMPT,
            """Analyze this crop image. Return a single JSON object only, with every
required key: is_target_crop, crop_name, disease_name, confidence_score,
severity_level, affected_area_percentage, visual_symptoms, remedies,
marathi_translation, escalate_to_expert, and helpline_number.

remedies must contain organic, chemical, and preventive_measures arrays.
marathi_translation must contain crop_name, disease_name, summary,
organic_remedies, chemical_remedies, and preventive_measures.
Use the exact crop values Cotton, Soybean, Onion, Other, or Not a plant.
For another crop or a non-plant, set is_target_crop to false, severity_level
to None, affected_area_percentage to 0, and all remedy arrays to empty.
Return no Markdown and no text outside the JSON object.""",
            types.Part.from_bytes(data=image, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response")

    return response.text


async def call_gemini(image: bytes, mime_type: str) -> dict[str, Any]:
    """Call Gemini with JSON mode enabled safely inside an async loop."""
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured")

    # Run the blocking Gemini SDK call in a separate thread
    raw_text = await asyncio.to_thread(_sync_gemini_call, image, mime_type)
    return extract_json(raw_text)


async def call_openai(image: bytes, mime_type: str) -> dict[str, Any]:
    """Call GPT-4o-mini using a data URL and JSON object mode."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    from openai import AsyncOpenAI
    data_url = f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(model=settings.openai_model, temperature=0, response_format={"type": "json_object"}, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": [{"type": "text", "text": "Analyze this image."}, {"type": "image_url", "image_url": {"url": data_url}}]}])
    if not response.choices[0].message.content:
        raise ValueError("Provider response was empty")
    return extract_json(response.choices[0].message.content)


app = FastAPI(title="Maharashtra Crop Disease API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(HTTPException)
async def handle_http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return error_response(exc.status_code, "request_error", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
    return error_response(422, "validation_error", "Upload an image using the 'image' form field.")


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return readiness information without exposing secrets."""
    return {"status": "ok", "target_crops": list(TARGET_CROPS), "vision_provider": settings.vision_provider}


@app.post("/api/v1/diagnose", response_model=DiagnosisResponse, responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def diagnose(image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP; 10 MB maximum")]) -> DiagnosisResponse:
    """Validate the upload, invoke the configured vision model, and validate its JSON."""
    image_bytes, mime_type = await read_and_verify_image(image)
    try:
        result = await (call_gemini(image_bytes, mime_type) if settings.vision_provider == "gemini" else call_openai(image_bytes, mime_type))
        return DiagnosisResponse.model_validate(result)
    except RuntimeError as exc:
        logger.error("Vision service configuration problem: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Diagnosis service is not configured.") from exc
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Invalid vision response: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Diagnosis service returned an invalid response; retry with a clearer image.") from exc
    except Exception as exc:
        logger.exception("Vision provider request failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Diagnosis service is temporarily unavailable.") from exc


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(Path(__file__).resolve().parent / "index.html")

