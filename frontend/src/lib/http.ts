import { supabase } from './supabase';
import { env } from './env';

/**
 * Retrieves the authorization headers including the bearer JWT token from the active session.
 * Attempts a session refresh if the initial getSession() returns null.
 */
export async function getAuthHeaders(): Promise<HeadersInit> {
  let { data: { session } } = await supabase.auth.getSession();

  // If no session, attempt a silent refresh before failing
  if (!session) {
    const { data: refreshData } = await supabase.auth.refreshSession();
    session = refreshData?.session ?? null;
  }

  const token = session?.access_token;

  return {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
  };
}

/**
 * Performs an HTTP request to the backend API, automatically injecting the Bearer Authorization token.
 */
export async function apiRequest(path: string, options: RequestInit = {}): Promise<Response> {
  const url = `${env.API_BASE_URL}${path}`;
  const authHeaders = await getAuthHeaders();
  
  return fetch(url, {
    ...options,
    headers: {
      ...authHeaders,
      ...options.headers,
    },
  });
}

