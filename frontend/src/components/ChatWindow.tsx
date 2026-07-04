import { useEffect, useState, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { useChat, type UIMessage } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { supabase } from '@/lib/supabase';
import { getThreadMessages, type CitationResponse } from '@/lib/api';
import { env } from '@/lib/env';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { 
  Sparkles, 
  User, 
  AlertCircle, 
  Loader2, 
  Paperclip,
  Search,
  ArrowRight,
  CheckCircle2,
  X
} from 'lucide-react';
import { cn } from '@/lib/utils';
import SourcePassagePanel from './SourcePassagePanel';


interface ChatWindowProps {
  threadId: string;
  threadTitle: string;
  onUploadStart?: (filename: string) => void;
}

export default function ChatWindow({ threadId, threadTitle, onUploadStart }: ChatWindowProps) {
  const [initialMessages, setInitialMessages] = useState<UIMessage[] | null>(null);
  const [initialCitations, setInitialCitations] = useState<Record<string, CitationResponse[]> | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function loadChatData() {
      try {
        setLoading(true);
        setError(null);

        // Get session token
        const { data: { session } } = await supabase.auth.getSession();
        if (!session) {
          throw new Error('No active Supabase session found.');
        }
        
        if (isMounted) {
          setToken(session.access_token);
        }

        // Fetch thread messages
        const history = await getThreadMessages(threadId);
        
        if (isMounted) {
          const formatted: UIMessage[] = history.map((msg) => ({
            id: msg.id,
            role: msg.role as 'user' | 'assistant',
            parts: [{ type: 'text', text: msg.content }],
          }));

          const initialCits: Record<string, CitationResponse[]> = {};
          history.forEach((msg) => {
            if (msg.citations && msg.citations.length > 0) {
              initialCits[msg.id] = msg.citations;
            }
          });

          setInitialMessages(formatted);
          setInitialCitations(initialCits);
        }
      } catch (err) {
        console.error('Error loading chat data:', err);
        if (isMounted) {
          const errorMessage = err instanceof Error ? err.message : 'Failed to load thread history.';
          setError(errorMessage);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    loadChatData();

    return () => {
      isMounted = false;
    };
  }, [threadId]);

  if (loading) {
    return (
      <div className="flex h-full flex-col items-center justify-center bg-[#030712] text-slate-400 gap-3">
        <Loader2 className="h-7 w-7 animate-spin text-blue-400" />
        <p className="text-sm font-medium text-slate-500">Loading conversation history...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center bg-[#030712] px-4 text-center">
        <div className="rounded-full bg-red-500/10 border border-red-500/20 p-3 text-red-400 mb-4">
          <AlertCircle className="h-7 w-7" />
        </div>
        <h3 className="text-base font-semibold text-slate-200">Failed to load conversation</h3>
        <p className="text-sm text-slate-500 mt-1 max-w-md">{error}</p>
      </div>
    );
  }

  if (!token || initialMessages === null || initialCitations === null) {
    return (
      <div className="flex h-full flex-col items-center justify-center bg-[#030712] text-slate-500">
        <AlertCircle className="h-7 w-7 mb-2" />
        <p className="text-sm">Session unauthorized or thread data unavailable.</p>
      </div>
    );
  }

  return (
    <ChatArea 
      threadId={threadId} 
      threadTitle={threadTitle} 
      initialMessages={initialMessages} 
      token={token} 
      initialCitations={initialCitations}
      onUploadStart={onUploadStart}
    />
  );
}

interface ChatAreaProps {
  threadId: string;
  threadTitle: string;
  initialMessages: UIMessage[];
  token: string;
  initialCitations: Record<string, CitationResponse[]>;
  onUploadStart?: (filename: string) => void;
}

function ChatArea({ threadId, threadTitle, initialMessages, token, initialCitations, onUploadStart }: ChatAreaProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const location = useLocation();
  const initialPromptRun = useRef(false);
  const [input, setInput] = useState('');

  // Upload state
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadState, setUploadState] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [uploadFileName, setUploadFileName] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Only PDF files are supported.');
      setUploadState('error');
      return;
    }
    setUploadFileName(file.name);
    setUploadState('uploading');
    setUploadError(null);

    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) throw new Error('Not authenticated');

      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${env.API_BASE_URL}/chat/upload`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${session.access_token}` },
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data?.detail || `Upload failed (${res.status})`);
      }

      setUploadState('success');
      // Notify parent (Dashboard) to start tracking this doc
      onUploadStart?.(file.name);
      // Auto-clear success banner after 4s
      setTimeout(() => { setUploadState('idle'); setUploadFileName(null); }, 4000);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed.');
      setUploadState('error');
    } finally {
      // Reset file input so the same file can be re-uploaded
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Citations & context state
  const [messageCitations, setMessageCitations] = useState<Record<string, CitationResponse[]>>(initialCitations);
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<string | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  
  const messagesRef = useRef<UIMessage[]>([]);
  
  const { messages, sendMessage, status } = useChat({
    messages: initialMessages,
    transport: new DefaultChatTransport({
      api: `${env.API_BASE_URL}/chat/stream`,
      body: {
        thread_id: threadId,
      },
      headers: {
        Authorization: `Bearer ${token}`,
      },
      prepareSendMessagesRequest: ({ messages, body, headers, credentials, api }) => {
        const formattedMessages = messages.map(msg => {
          const textContent = msg.parts
            .filter(p => p.type === 'text')
            .map(p => (p as { type: 'text'; text: string }).text)
            .join('');
          return {
            role: msg.role,
            content: textContent
          };
        });
        return {
          api,
          headers,
          credentials,
          body: {
            ...body,
            messages: formattedMessages
          }
        };
      }
    }),
    onData: (dataPart) => {
      if (dataPart.type === 'data-retrieved_context') {
        setPipelineStatus('Synthesizing grounded response...');
      } else if (dataPart.type === 'data-citations') {
        const citations = (dataPart.data as any)?.citations;
        // Associate citations with the last assistant message using messagesRef
        const lastAssistantMsg = [...messagesRef.current].reverse().find(m => m.role === 'assistant');
        if (lastAssistantMsg && citations) {
          setMessageCitations(prev => ({
            ...prev,
            [lastAssistantMsg.id]: citations
          }));
        }
        setPipelineStatus(null);
      }
    },
    onError: (err: Error) => {
      console.error('Streaming error:', err);
      let cleanMsg = err.message || 'Network error encountered during stream.';
      if (cleanMsg.includes('Grounding Error:')) {
        cleanMsg = cleanMsg.replace(/^Error:\s*/i, '');
      }
      setStreamError(cleanMsg);
      setPipelineStatus(null);
    }
  });

  // Keep messagesRef updated to prevent stale closures in onData
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // Sync statuses with useChat state transitions
  useEffect(() => {
    if (status === 'submitted') {
      setPipelineStatus('Retrieving context...');
      setStreamError(null);
    } else if (status === 'streaming') {
      setPipelineStatus('Generating response...');
    }
  }, [status]);

  // Automatically execute the initial prompt suggestion if present in routing state
  useEffect(() => {
    if (location.state?.initialPrompt && !initialPromptRun.current && messages.length === 0) {
      initialPromptRun.current = true;
      // Clear location state to avoid double execution on reload
      window.history.replaceState({}, document.title);
      sendMessage({
        text: location.state.initialPrompt,
      });
      setStreamError(null);
      setPipelineStatus('Retrieving context...');
    }
  }, [location.state, messages, sendMessage]);

  // Scroll to bottom when messages or loading state changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, status]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || status === 'submitted' || status === 'streaming') return;
    sendMessage({ text: input });
    setInput('');
    setStreamError(null);
    setPipelineStatus('Retrieving context...');
  };


  const isModelGenerating = status === 'submitted' || status === 'streaming';

  return (
    <div className="flex h-full w-full bg-[#030712] text-slate-100 overflow-hidden">
      {/* Main Chat Viewport */}
      <div className="flex-1 flex flex-col h-full bg-[#030712] text-slate-100 relative overflow-hidden border-r border-slate-900/40">
        {/* Background radial highlight */}
        <div className="absolute top-0 right-1/4 h-[300px] w-[500px] rounded-full bg-blue-600/5 blur-[100px] pointer-events-none"></div>

        {/* Chat Header */}
        <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-slate-900 bg-[#030712]/50 backdrop-blur-xl px-6">
          <div className="flex items-center gap-3">
            <h1 className="text-base font-semibold text-slate-100">
              {threadTitle || "Untitled Workspace"}
            </h1>
          </div>
        </header>

        {/* Messages Viewport */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6 scrollbar-none">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center max-w-md mx-auto text-center space-y-4 py-12">
              <div className="h-10 w-10 rounded-xl bg-blue-950/20 border border-blue-500/25 flex items-center justify-center text-blue-400">
                <Sparkles className="h-5 w-5" />
              </div>
              <div className="space-y-1.5">
                <h2 className="text-sm font-semibold text-slate-200">Session Initialized</h2>
                <p className="text-xs text-slate-450 leading-relaxed">
                  Submit a query to parse and analyze filing data.
                </p>
              </div>
            </div>
          ) : (
            messages.map((message) => {
              const isUser = message.role === 'user';
              const textContent = message.parts
                .filter((p) => p.type === 'text')
                .map((p) => (p as { type: 'text'; text: string }).text)
                .join('');

              const citations = messageCitations[message.id] || [];

              return (
                <div
                  key={message.id}
                  className={cn(
                    "flex w-full gap-4 max-w-3xl animate-fade-in",
                    isUser ? "ml-auto flex-row-reverse" : "mr-auto"
                  )}
                >
                  {/* Avatar */}
                  <div className={cn(
                    "flex h-9 w-9 shrink-0 select-none items-center justify-center rounded-xl border text-xs font-semibold shadow-sm",
                    isUser 
                      ? "bg-slate-900 border-slate-800/80 text-slate-400" 
                      : "bg-blue-950/20 border-blue-500/25 text-blue-400"
                  )}>
                    {isUser ? <User className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
                  </div>

                  {/* Message Bubble */}
                  <div className={cn(
                    "flex flex-col gap-1.5 max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-md transition-all duration-200",
                    isUser 
                      ? "bg-blue-600/10 border border-blue-500/20 text-slate-100 rounded-tr-none" 
                      : "bg-[#070b16]/50 border border-slate-800/60 backdrop-blur-md text-slate-200 rounded-tl-none"
                  )}>
                    {isUser ? (
                      <div className="whitespace-pre-wrap leading-relaxed">{textContent}</div>
                    ) : (
                      <div className="leading-relaxed markdown-body">
                        <AssistantMessage
                          text={textContent}
                          citations={citations}
                          onCitationClick={(chunkId) => setSelectedChunkId(chunkId)}
                        />
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}

          {/* Grounding Error Display */}
          {streamError && (
            <div className="flex w-full gap-4 max-w-3xl mr-auto animate-fade-in">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-red-500/25 bg-red-950/10 text-red-400">
                <AlertCircle className="h-4 w-4" />
              </div>
              <div className="flex flex-col gap-1.5 rounded-2xl rounded-tl-none bg-red-950/20 border border-red-500/20 p-4 text-sm shadow-md text-red-400">
                <span className="font-semibold text-xxs tracking-wider uppercase font-mono">Grounding Policy Blocked</span>
                <p className="text-xs leading-relaxed mt-0.5">{streamError}</p>
              </div>
            </div>
          )}

          {/* Streaming/Loading Indicator */}
          {isModelGenerating && (
            <div className="flex w-full gap-4 max-w-3xl mr-auto animate-pulse">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-blue-500/20 bg-blue-950/10 text-blue-400">
                <Sparkles className="h-4 w-4 animate-spin-slow" />
              </div>
              <div className="flex flex-col gap-2 rounded-2xl rounded-tl-none bg-[#070b16]/50 border border-slate-800/60 p-4 text-sm shadow-md text-slate-400 min-w-[200px]">
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="h-2 w-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="h-2 w-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
                {pipelineStatus && (
                  <span className="text-[10px] font-mono text-slate-500 mt-1 uppercase tracking-wider">{pipelineStatus}</span>
                )}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Drawer */}
        <footer className="sticky bottom-0 z-20 border-t border-slate-900/40 bg-[#030712]/80 backdrop-blur-md px-6 py-4">
          {/* Hidden PDF file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            className="hidden"
            onChange={handleFileUpload}
          />
          <form onSubmit={handleSubmit} className="max-w-2xl mx-auto">

            {/* Upload status banner */}
            {uploadState !== 'idle' && (
              <div className={cn(
                "flex items-center justify-between gap-3 rounded-xl px-3 py-2 mb-3 text-xs font-semibold border transition-all",
                uploadState === 'uploading' && "bg-blue-950/20 border-blue-500/20 text-blue-300",
                uploadState === 'success'  && "bg-green-950/20 border-green-500/20 text-green-300",
                uploadState === 'error'    && "bg-red-950/20 border-red-500/20 text-red-300",
              )}>
                <div className="flex items-center gap-2 min-w-0">
                  {uploadState === 'uploading' && <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />}
                  {uploadState === 'success'  && <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />}
                  {uploadState === 'error'    && <AlertCircle className="h-3.5 w-3.5 shrink-0" />}
                  <span className="truncate">
                    {uploadState === 'uploading' && `Uploading "${uploadFileName}"...`}
                    {uploadState === 'success'  && `"${uploadFileName}" received — ingesting in background.`}
                    {uploadState === 'error'    && (uploadError || 'Upload failed.')}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => { setUploadState('idle'); setUploadFileName(null); setUploadError(null); }}
                  className="shrink-0 opacity-60 hover:opacity-100 transition-opacity"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            )}

            <div className="bg-[#070b16]/90 border border-slate-800/80 rounded-2xl p-3.5 shadow-2xl focus-within:border-blue-500/50 focus-within:ring-2 focus-within:ring-blue-500/10 transition-all duration-300 relative text-left">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask the Copilot about company performance or files..."
                className="w-full bg-transparent border-0 outline-none resize-none text-slate-200 placeholder-slate-500 text-sm h-14 focus:ring-0 px-1"
                disabled={isModelGenerating}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
              />
              
              {/* Input action toolbar */}
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-900/80">
                <div className="flex items-center gap-2">
                  <span className="rounded-xl px-3 py-1 text-[10px] font-semibold flex items-center gap-1.5 bg-blue-600 text-slate-50 shadow-md shadow-blue-500/10">
                    <Search className="h-2.5 w-2.5" />
                    <span>Search Filings</span>
                  </span>
                </div>
                
                <div className="flex items-center gap-2 text-slate-500">
                  <button
                    type="button"
                    onClick={(e) => { e.preventDefault(); e.stopPropagation(); fileInputRef.current?.click(); }}
                    disabled={uploadState === 'uploading'}
                    className="hover:text-blue-400 transition-colors p-1 disabled:opacity-40 cursor-pointer"
                    title="Upload PDF document"
                  >
                    {uploadState === 'uploading' ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" />
                    ) : (
                      <Paperclip className="h-3.5 w-3.5" />
                    )}
                  </button>
                  <button 
                    type="submit"
                    disabled={!input.trim() || isModelGenerating}
                    className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white rounded-lg p-1.5 shadow transition-colors cursor-pointer"
                  >
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            </div>
          </form>
        </footer>
      </div>

      {/* Audit sidebar for chunk inspect */}
      {selectedChunkId && (
        <SourcePassagePanel 
          chunkId={selectedChunkId} 
          onClose={() => setSelectedChunkId(null)} 
        />
      )}
    </div>
  );
}

// ─── Citation-aware Markdown renderer ────────────────────────────────────────

interface AssistantMessageProps {
  text: string;
  citations: CitationResponse[];
  onCitationClick: (chunkId: string) => void;
}

/** Citation regex — matches [filename.pdf, Page N] supporting optional spaces, bold asterisks, and varied punctuation */
const CITATION_REGEX = /\[\s*\*?\*?\s*([^,\]\*]+?)\s*\*?\*?\s*,\s*\*?\*?\s*[pP]age\s*\*?\*?\s*(\d+)\s*\*?\*?\s*\]/g;

function buildDisplayLabel(filename: string, pageNum: number): string {
  let company = filename.replace(/_.*$/, '');
  if (company.toUpperCase() === 'RELIANCE') company = 'Reliance';
  else if (company.toUpperCase() === 'TCS') company = 'TCS';
  const yearMatch = filename.match(/\d{4}/);
  const year = yearMatch ? `'${yearMatch[0].slice(-2)}` : '';
  return `${company} ${year}, p. ${pageNum}`;
}

/**
 * Renders an assistant message with:
 *  - Full GFM markdown (bold, italic, tables, lists, code, etc.)
 *  - Citation markers replaced by interactive citation buttons
 */
function AssistantMessage({ text, citations, onCitationClick }: AssistantMessageProps) {
  // Pre-process: replace citation markers with markdown link placeholders
  // e.g. [TCS_2024.pdf, Page 12] → [TCS '24, p. 12](cit://0)
  const citationStore: Record<string, { chunkId: string | null; label: string }> = {};
  let citIdx = 0;

  // Normalize thick full-width brackets to standard brackets
  const normalizedText = text.replace(/【/g, '[').replace(/】/g, ']');

  const processedText = normalizedText.replace(CITATION_REGEX, (_match, filename, pageStr) => {
    const cleanFilename = filename.trim();
    const pageNum = parseInt(pageStr.trim(), 10);
    const label = buildDisplayLabel(cleanFilename, pageNum);
    const key = `cit-${citIdx++}`;

    const found = citations?.find(
      c => c.filename.toLowerCase() === cleanFilename.toLowerCase() && c.page_number === pageNum
    );
    citationStore[key] = { chunkId: found?.chunk_id ?? null, label };

    return `[${label}](https://citation.local/${key})`;
  });

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        // Paragraphs
        p: ({ children }) => (
          <p className="mb-2 last:mb-0 text-sm text-slate-200 leading-relaxed">{children}</p>
        ),
        // Headings
        h1: ({ children }) => (
          <h1 className="text-base font-bold text-slate-100 mt-4 mb-2 first:mt-0">{children}</h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-sm font-bold text-slate-100 mt-3 mb-1.5 first:mt-0">{children}</h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-sm font-semibold text-slate-200 mt-2 mb-1 first:mt-0">{children}</h3>
        ),
        // Emphasis
        strong: ({ children }) => (
          <strong className="font-semibold text-slate-100">{children}</strong>
        ),
        em: ({ children }) => (
          <em className="italic text-slate-300">{children}</em>
        ),
        // Lists
        ul: ({ children }) => (
          <ul className="list-disc list-outside ml-4 space-y-0.5 mb-2 text-slate-300">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal list-outside ml-4 space-y-0.5 mb-2 text-slate-300">{children}</ol>
        ),
        li: ({ children }) => (
          <li className="text-sm leading-relaxed">{children}</li>
        ),
        // Code
        code: ({ className, children }) => {
          const isBlock = Boolean(className);
          const rawText = typeof children === 'string' ? children : String(children || '');
          
          // Render citations interactively even inside code blocks / preformatted tables
          const processedChildren = (() => {
            const LINK_REGEX = /\[([^\]]+)\]\(https:\/\/citation.local\/(cit-\d+)\)/g;
            const parts: React.ReactNode[] = [];
            let lastIndex = 0;
            let match;
            
            while ((match = LINK_REGEX.exec(rawText)) !== null) {
              const matchIndex = match.index;
              const label = match[1];
              const key = match[2];
              
              if (matchIndex > lastIndex) {
                parts.push(rawText.substring(lastIndex, matchIndex));
              }
              
              const cit = citationStore[key];
              if (cit?.chunkId) {
                parts.push(
                  <button
                    key={matchIndex}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      onCitationClick(cit.chunkId!);
                    }}
                    className="inline-flex items-center gap-1 px-1 py-0.5 rounded bg-blue-500/10 border border-blue-500/30 text-blue-300 font-mono text-[9px] font-semibold hover:bg-blue-500/20 active:scale-95 transition-all mx-0.5 align-middle cursor-pointer"
                    title={`View source: ${cit.label}`}
                  >
                    {label}
                  </button>
                );
              } else {
                parts.push(
                  <span
                    key={matchIndex}
                    className="inline-flex items-center gap-1 px-1 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono text-[9px] mx-0.5 align-middle select-none"
                  >
                    {label}
                  </span>
                );
              }
              
              lastIndex = LINK_REGEX.lastIndex;
            }
            
            if (lastIndex < rawText.length) {
              parts.push(rawText.substring(lastIndex));
            }
            
            return parts.length > 0 ? parts : children;
          })();

          // Check if children contain any citation element (resolved button or unresolved span)
          const hasCitation = Array.isArray(processedChildren)
            ? processedChildren.some(child => typeof child === 'object' && child !== null)
            : typeof processedChildren === 'object' && processedChildren !== null;

          if (isBlock) {
            return (
              <pre className="bg-slate-900/80 border border-slate-800 rounded-lg p-3 overflow-x-auto mb-2">
                <code className={cn("font-mono text-xs text-blue-300", className)}>
                  {processedChildren}
                </code>
              </pre>
            );
          }

          if (hasCitation) {
            return <>{processedChildren}</>;
          }

          return (
            <code className="font-mono text-xs bg-slate-900 text-blue-300 px-1 py-0.5 rounded">
              {processedChildren}
            </code>
          );
        },
        pre: ({ children }) => <>{children}</>,
        // Blockquote
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-blue-500/40 pl-3 my-2 text-slate-400 italic">
            {children}
          </blockquote>
        ),
        // Tables
        table: ({ children }) => (
          <div className="overflow-x-auto mb-2">
            <table className="w-full text-xs border-collapse">{children}</table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-slate-900/60 text-slate-300 font-semibold">{children}</thead>
        ),
        tbody: ({ children }) => <tbody className="text-slate-400">{children}</tbody>,
        tr: ({ children }) => <tr className="border-b border-slate-800/60">{children}</tr>,
        th: ({ children }) => <th className="px-3 py-1.5 text-left">{children}</th>,
        td: ({ children }) => <td className="px-3 py-1.5">{children}</td>,
        // Horizontal rule
        hr: () => <hr className="border-slate-800 my-3" />,
        // Links — intercept citation links
        a: ({ href, children }) => {
          if (href?.startsWith('https://citation.local/')) {
            const key = href.replace('https://citation.local/', '');
            const cit = citationStore[key];
            if (cit?.chunkId) {
              return (
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onCitationClick(cit.chunkId!);
                  }}
                  className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-blue-500/10 border border-blue-500/30 text-blue-300 font-mono text-[10px] font-semibold hover:bg-blue-500/20 active:scale-95 transition-all mx-0.5 align-middle cursor-pointer"
                  title={`View source: ${cit.label}`}
                >
                  {children}
                </button>
              );
            }
            // Unresolved citation
            return (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono text-[10px] mx-0.5 align-middle select-none">
                {children}
              </span>
            );
          }
          return (
            <a
              href={href}
              target="_blank"
              rel="noreferrer"
              className="text-blue-400 underline hover:text-blue-300 transition-colors"
            >
              {children}
            </a>
          );
        },
      }}
    >
      {processedText}
    </ReactMarkdown>
  );
}
