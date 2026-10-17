import { apiRequest, handleApiResponse } from './http';

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

export interface UserDocument {
  id: string;
  filename: string;
  created_at: string;
}

export interface PublicDocument {
  id: string;
  filename: string;
  ticker: string;
  year: number;
}

export interface UploadDocumentResponse {
  status: string;
  filename: string;
  message: string;
}

export async function testAuth(): Promise<TestAuthResponse> {
  const response = await apiRequest('/auth/test-me');
  return handleApiResponse<TestAuthResponse>(response);
}

export async function listThreads(): Promise<ThreadResponse[]> {
  const response = await apiRequest('/chat/threads');
  return handleApiResponse<ThreadResponse[]>(response);
}

export async function createThread(title: string): Promise<ThreadResponse> {
  const response = await apiRequest('/chat/threads', {
    method: 'POST',
    body: JSON.stringify({ title }),
  });
  return handleApiResponse<ThreadResponse>(response);
}

export async function getThreadMessages(threadId: string): Promise<MessageResponse[]> {
  const response = await apiRequest(`/chat/threads/${threadId}/messages`);
  return handleApiResponse<MessageResponse[]>(response);
}

export async function deleteThread(threadId: string): Promise<void> {
  const response = await apiRequest(`/chat/threads/${threadId}`, {
    method: 'DELETE',
  });
  return handleApiResponse<void>(response);
}

export async function getChunkDetails(chunkId: string): Promise<ChunkDetailsResponse> {
  const response = await apiRequest(`/chat/chunks/${chunkId}`);
  return handleApiResponse<ChunkDetailsResponse>(response);
}

export async function listUserDocuments(): Promise<UserDocument[]> {
  const response = await apiRequest('/chat/documents');
  return handleApiResponse<UserDocument[]>(response);
}

export async function listPublicDocuments(): Promise<PublicDocument[]> {
  const response = await apiRequest('/chat/documents/public');
  return handleApiResponse<PublicDocument[]>(response);
}

export async function deleteDocument(docId: string): Promise<void> {
  const response = await apiRequest(`/chat/documents/${docId}`, {
    method: 'DELETE',
  });
  return handleApiResponse<void>(response);
}

/**
 * Uploads a PDF document to the backend for background parsing, chunking, and embedding.
 * Automatically validates PDF extension and leverages handleApiResponse for rich error details.
 */
export async function uploadDocument(file: File): Promise<UploadDocumentResponse> {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    throw new Error('Only PDF files are supported.');
  }

  const formData = new FormData();
  formData.append('file', file);

  const response = await apiRequest('/chat/upload', {
    method: 'POST',
    body: formData,
  });

  return handleApiResponse<UploadDocumentResponse>(response);
}
