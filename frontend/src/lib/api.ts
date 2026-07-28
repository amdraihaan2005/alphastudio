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

export async function testAuth(): Promise<TestAuthResponse> {
  const response = await apiRequest('/auth/test-me');
  if (!response.ok) {
    throw new Error(`Authentication check failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function listThreads(): Promise<ThreadResponse[]> {
  const response = await apiRequest('/chat/threads');
  if (!response.ok) {
    throw new Error(`Failed to fetch threads: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

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

export async function getThreadMessages(threadId: string): Promise<MessageResponse[]> {
  const response = await apiRequest(`/chat/threads/${threadId}/messages`);
  if (!response.ok) {
    throw new Error(`Failed to fetch messages: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function deleteThread(threadId: string): Promise<void> {
  const response = await apiRequest(`/chat/threads/${threadId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error(`Failed to delete thread: ${response.status} ${response.statusText}`);
  }
}

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

export async function listUserDocuments(): Promise<UserDocument[]> {
  const response = await apiRequest('/chat/documents');
  if (!response.ok) {
    throw new Error(`Failed to fetch user documents: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function listPublicDocuments(): Promise<{ id: string; filename: string; ticker: string; year: number }[]> {
  const response = await apiRequest('/chat/documents/public');
  if (!response.ok) {
    throw new Error(`Failed to fetch public documents: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function deleteDocument(docId: string): Promise<void> {
  const response = await apiRequest(`/chat/documents/${docId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error(`Failed to delete document: ${response.status} ${response.statusText}`);
  }
}
