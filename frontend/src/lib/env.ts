export const env = {
  API_BASE_URL: (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000',
  SUPABASE_URL: import.meta.env.VITE_SUPABASE_URL as string,
  SUPABASE_ANON_KEY: import.meta.env.VITE_SUPABASE_ANON_KEY as string,
};

// Validate that required variables are defined
if (!env.SUPABASE_URL || !env.SUPABASE_ANON_KEY) {
  throw new Error(
    "Missing required environment variables: VITE_SUPABASE_URL and/or VITE_SUPABASE_ANON_KEY must be defined."
  );
}
