import { supabase } from './supabase';
import { env } from './env';

/**
 * Returns a valid Supabase access token, attempting session refresh if needed.
 */
export async function getValidAccessToken(): Promise<string | null> {
  let { data: { session } } = await supabase.auth.getSession();

  if (!session) {
    const { data: refreshData } = await supabase.auth.refreshSession();
    session = refreshData?.session ?? null;
  }

  return session?.access_token ?? null;
}

/**
 * Returns authorization and optional content-type headers for backend requests.
 */
export async function getAuthHeaders(includeContentType = true): Promise<Record<string, string>> {
  const token = await getValidAccessToken();

  return {
    ...(includeContentType ? { 'Content-Type': 'application/json' } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

/**
 * Robust API request client with automatic token attachment, multipart support,
 * and base URL routing.
 */
export async function apiRequest(path: string, options: RequestInit = {}): Promise<Response> {
  const url = `${env.API_BASE_URL}${path}`;
  const isFormData = typeof FormData !== 'undefined' && options.body instanceof FormData;
  const authHeaders = await getAuthHeaders(!isFormData);

  return fetch(url, {
    ...options,
    headers: {
      ...authHeaders,
      ...options.headers,
    },
  });
}

/**
 * Centralized response unwrap handler that extracts backend FastAPI error details
 * (e.g. { "detail": "..." }) instead of throwing vague HTTP status codes.
 */
export async function handleApiResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = `Request failed (${response.status} ${response.statusText})`;
    try {
      const data = await response.json();
      if (typeof data?.detail === 'string') {
        errorDetail = data.detail;
      } else if (Array.isArray(data?.detail)) {
        errorDetail = data.detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join(', ');
      } else if (data?.message) {
        errorDetail = data.message;
      }
    } catch {
      // Body was not JSON
    }
    throw new Error(errorDetail);
  }

  if (response.status === 204) {
    return undefined as unknown as T;
  }

  return response.json();
}
