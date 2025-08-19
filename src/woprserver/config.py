from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class ContactInfo(BaseModel):
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    email: EmailStr

class LicenseInfo(BaseModel):
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)

class BrandingConfig(BaseModel):
    title: str = "WOPRserver Data Plane"
    version: str = "9.9"
    description: str = "🌌 WOPRserver — Parallel Inference Engine at planet-destroying speed."
    contact: ContactInfo = ContactInfo(name="WOPRserver Team", url="https://woprserver.ai", email="contact@woprserver.ai")
    license: LicenseInfo = LicenseInfo(name="Cosmic License v1.0", url="https://woprserver.ai/license")

class MergeConfig(BaseModel):
    src_namespace: str = "woprserver"
    src_subpkg: str = "parallel"
    dst_namespace: str = "mlserver"
    dst_subpkg: str = "parallel"
    modules: list[str] = ["dispatcher","errors","logging","messages","model","pool","registry","utils","worker"]
    overwrite_existing: bool = True
    copy_private: bool = False

class LoggingConfig(BaseModel):
    level: str = "INFO"
    use_color: bool | None = None
    rename_mlserver_loggers: bool = True
    neutralize_mlserver: bool = True
    disarm_uvicorn: bool = True
    guard_future_handlers: bool = True
    capture_warnings: bool = True

class WoprConfig(BaseModel):
    branding: BrandingConfig = BrandingConfig()
    merge: MergeConfig = MergeConfig()
    logging: LoggingConfig = LoggingConfig()

def default_config() -> WoprConfig:
    return WoprConfig()