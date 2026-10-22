from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_async_database_url

# Alembic has a synchronous engine of its own (alembic/env.py); the application
# only uses the async one below.
_async_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_async_engine() -> AsyncEngine:
    """Create the engine lazily; no connection is opened until first use."""
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            get_async_database_url(), pool_pre_ping=True
        )
    return _async_engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_async_engine(), expire_on_commit=False
        )
    return _async_session_factory


async def dispose_async_engine() -> None:
    global _async_engine, _async_session_factory
    if _async_engine is not None:
        await _async_engine.dispose()
    _async_engine = None
    _async_session_factory = None
