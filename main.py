from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.config import settings
from src.payment_tracker import PaymentTracker
from src.schemas import ProcessPathRequest, ProcessResponse
from src.utils import parse_reference_date


app = FastAPI(
    title="Payment Tracking API",
    version="1.0.0",
    description="Process rent schedule and bank statement Excel files into a payment status report.",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "defaults": {
            "arenda": str(settings.input_dir / settings.default_arenda_file),
            "bank": str(settings.input_dir / settings.default_bank_file),
            "report": str(settings.output_dir / settings.default_report_file),
        },
    }


@app.post("/process/path", response_model=ProcessResponse)
def process_path(request: ProcessPathRequest) -> ProcessResponse:
    try:
        today = parse_reference_date(request.today)
        tracker = PaymentTracker(grace_days=request.grace_days)
        result = tracker.process(
            arenda_path=request.arenda_path,
            bank_path=request.bank_path,
            report_path=request.report_path,
            today=today,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ProcessResponse(
        report_path=str(result.report_path),
        summary=result.summary,
        preview=result.report.head(10).fillna("").to_dict(orient="records"),
    )


@app.post("/process/upload", response_model=ProcessResponse)
async def process_upload(
    arenda_file: UploadFile = File(...),
    bank_file: UploadFile = File(...),
    today: str | None = Form(default=None),
    grace_days: int = Form(default=3),
) -> ProcessResponse:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        arenda_path = tmp_dir / "arenda.xlsx"
        bank_path = tmp_dir / "print.xlsx"
        report_path = settings.output_dir / settings.default_report_file

        with arenda_path.open("wb") as f:
            shutil.copyfileobj(arenda_file.file, f)
        with bank_path.open("wb") as f:
            shutil.copyfileobj(bank_file.file, f)

        try:
            ref_date = parse_reference_date(today)
            tracker = PaymentTracker(grace_days=grace_days)
            result = tracker.process(
                arenda_path=arenda_path,
                bank_path=bank_path,
                report_path=report_path,
                today=ref_date,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ProcessResponse(
        report_path=str(result.report_path),
        summary=result.summary,
        preview=result.report.head(10).fillna("").to_dict(orient="records"),
    )
