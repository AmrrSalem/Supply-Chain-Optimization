"""
Configuration Management Module
================================

This module provides configuration management for the Supply Chain Optimization system.
It supports different environments (development, production) and loads settings from
environment variables and configuration files.

Usage:
    from config import settings

    service_level = settings.SERVICE_LEVEL
    holding_cost_rate = settings.HOLDING_COST_RATE
"""

from config.settings import Settings

# Create global settings instance
settings = Settings()

__all__ = ['settings']
