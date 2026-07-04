from supabase import create_client, Client
from app.config import settings

# Public anon client - scoped to user permissions (abides by Row Level Security)
supabase: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_ANON_KEY
)

# Service role client - admin access (bypasses Row Level Security)
# DANGER: Only use this for backend scripts, ingestion, and system operations.
# NEVER expose this client or its keys to the frontend/client.
supabase_admin: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_SERVICE_ROLE_KEY
)
