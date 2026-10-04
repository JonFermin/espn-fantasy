"""Schema migrations: ``NNNN_name.sql`` files applied in order by ``fm.store.db.migrate``.

Each file is one migration, numbered from ``0001`` without gaps. A file must not manage transactions itself (the
runner wraps it in one) and is never edited once applied; schema changes go in the next file.
"""
