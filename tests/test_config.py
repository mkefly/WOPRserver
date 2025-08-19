from woprserver.config import (
    BrandingConfig,
    ContactInfo,
    LicenseInfo,
    LoggingConfig,
    MergeConfig,
    WoprConfig,
)


def test_pydantic_defaults_ok():
    cfg = WoprConfig()
    assert cfg.branding.title.startswith("WOPRserver")
    assert cfg.merge.overwrite_existing is True
    assert cfg.logging.rename_mlserver_loggers is True


def test_pydantic_custom_values_validate():
    branding = BrandingConfig(
        title="My Plane",
        version="1.0",
        description="desc",
        contact=ContactInfo(name="Me", url="https://x.y", email="me@x.y"),
        license=LicenseInfo(name="MIT", url="https://x.y/mit"),
    )
    merge = MergeConfig(modules=["worker"], overwrite_existing=False)
    logging_cfg = LoggingConfig(level="DEBUG", use_color=False)

    cfg = WoprConfig(branding=branding, merge=merge, logging=logging_cfg)
    assert cfg.branding.contact.email == "me@x.y"
    assert cfg.merge.modules == ["worker"]
    assert cfg.logging.level == "DEBUG"
