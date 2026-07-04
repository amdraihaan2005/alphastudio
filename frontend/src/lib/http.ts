import { supabase } from './supabase';
import { env } from './env';

export async function getAuthHeaders(): Promise<HeadersInit> {
  let { data: { session } } = await supabase.auth.getSession();

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

