from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import logging
import time
import json
import joblib
import numpy as np

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Setup Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup structured logging
logger = logging.getLogger("demo-log-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# FastAPI app
app = FastAPI()

# Load model
try:
    model = joblib.load("model.joblib")
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None

# Pydantic model for request validation
class TransactionFeatures(BaseModel):
    # Columns from creditcard.csv except 'Class'
    # Example: V1-V28 + Amount
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/live_check")
def live_check():
    with tracer.start_as_current_span("live_check"):
        logger.info("Live check endpoint called")
        return {"status": "alive"}

@app.get("/ready_check")
def ready_check():
    with tracer.start_as_current_span("ready_check"):
        status = "ready" if model else "model_not_loaded"
        logger.info(f"Ready check endpoint called: {status}")
        return {"status": status}

@app.post("/predict")
async def predict(features: TransactionFeatures, request: Request):
    with tracer.start_as_current_span("predict") as span:
        try:
            start_time = time.time()
            # Convert features to model input
            data = np.array([[getattr(features, col) for col in features.__fields__]])
            
            prediction = model.predict(data).tolist()
            probability = model.predict_proba(data).tolist()

            latency = time.time() - start_time

            # Add attributes to trace
            span.set_attribute("model", "creditcard-fraud")
            span.set_attribute("latency_ms", round(latency * 1000, 2))
            span.set_attribute("input", data.tolist())
            span.set_attribute("prediction", prediction)

            logger.info(f"Prediction: {prediction}, Latency: {latency:.3f}s")

            return {
                "prediction": prediction, 
                "probability": probability,
                "latency": latency
            }
        except Exception as e:
            span.record_exception(e)
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

