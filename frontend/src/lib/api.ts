import { apiRequest } from './http';

export interface TestAuthResponse {
  status: string;
  user_id: string;
  email: string;
}

export interface ThreadResponse {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
}

export interface CitationResponse {
  chunk_id: string;
  filename: string;
  page_number: number;
}

export interface MessageResponse {
  id: string;
  chat_thread_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  citations?: CitationResponse[];
}

export interface DocumentMetadata {
  id: string;
  filename: string;
  ticker: string;
  filing_type: string;
  year: number;
}

export interface ChunkDetailsResponse {
  id: string;
  text_content: string;
  page_number: number;
  section_name: string | null;
  document: DocumentMetadata;
  preceding_text: string | null;
  succeeding_text: string | null;
}


/**
 * Calls the backend `/auth/test-me` endpoint to verify authentication credentials.
 */
export async function testAuth(): Promise<TestAuthResponse> {
  const response = await apiRequest('/auth/test-me');
  if (!response.ok) {
    throw new Error(`Authentication check failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetch all chat threads belonging to the current user.
 */
export async function listThreads(): Promise<ThreadResponse[]> {
  const response = await apiRequest('/chat/threads');
  if (!response.ok) {
    throw new Error(`Failed to fetch threads: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

/**
 * Create a new chat thread.
 */
export async function createThread(title: string): Promise<ThreadResponse> {
  const response = await apiRequest('/chat/threads', {
    method: 'POST',
    body: JSON.stringify({ title }),
  });
  if (!response.ok) {
    throw new Error(`Failed to create thread: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

/**
 * Retrieve message history for a specific thread.
 */
export async function getThreadMessages(threadId: string): Promise<MessageResponse[]> {
  const response = await apiRequest(`/chat/threads/${threadId}/messages`);
  if (!response.ok) {
    throw new Error(`Failed to fetch messages: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

/**
 * Delete a specific chat thread.
 */
export async function deleteThread(threadId: string): Promise<void> {
  const response = await apiRequest(`/chat/threads/${threadId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error(`Failed to delete thread: ${response.status} ${response.statusText}`);
  }
}

/**
 * Retrieve document chunk details along with context from surrounding chunks.
 */
export async function getChunkDetails(chunkId: string): Promise<ChunkDetailsResponse> {
  const response = await apiRequest(`/chat/chunks/${chunkId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch chunk details: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export interface UserDocument {
  id: string;
  filename: string;
  created_at: string;
}

/**
 * List private PDF documents uploaded by the current user.
 */
export async function listUserDocuments(): Promise<UserDocument[]> {
  const response = await apiRequest('/chat/documents');
  if (!response.ok) {
    throw new Error(`Failed to fetch user documents: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

/**
 * List pre-loaded public documents (system-level filings, user_id=NULL).
 * These are always included in retrieval and cannot be deleted by users.
 */
export async function listPublicDocuments(): Promise<{ id: string; filename: string; ticker: string; year: number }[]> {
  const response = await apiRequest('/chat/documents/public');
  if (!response.ok) {
    throw new Error(`Failed to fetch public documents: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

/**
 * Permanently delete a private document uploaded by the current user.
 * Cascades to all associated chunks and citations on the backend.
 */
export async function deleteDocument(docId: string): Promise<void> {
  const response = await apiRequest(`/chat/documents/${docId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error(`Failed to delete document: ${response.status} ${response.statusText}`);
  }
}
