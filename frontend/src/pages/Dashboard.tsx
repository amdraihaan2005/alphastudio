import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { supabase } from '@/lib/supabase';
import {
  listThreads, createThread, deleteThread, type ThreadResponse,
  listUserDocuments, deleteDocument, listPublicDocuments, type UserDocument,
} from '@/lib/api';
import { Button } from '@/components/ui/button';
import ChatWindow from '@/components/ChatWindow';
import { env } from '@/lib/env';
import {
  LogOut, User, MessageSquare, Trash2, Plus, Loader2,
  AlertCircle, HelpCircle, Paperclip, ArrowRight, Search,
  Database, CheckCircle2, X, Lock, AlertTriangle,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// ─── Types ───────────────────────────────────────────────────────────────────

interface PendingDoc {
  filename: string;
  startTime: number;
  timedOut?: boolean;   // true when polling exceeded the window without finding the doc
}

interface PublicDoc {
  id: string;
  filename: string;
  ticker: string;
  year: number;
}

// ─── Constants ───────────────────────────────────────────────────────────────

/** Max time (ms) to wait for background ingestion before showing an error. */
const INGESTION_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes
/** Polling interval for checking whether a just-uploaded doc has been ingested. */
const POLLING_INTERVAL_MS = 4_000;

// ─── Component ───────────────────────────────────────────────────────────────

export default function Dashboard() {
  const { threadId } = useParams<{ threadId: string }>();
  const navigate = useNavigate();

  // Thread list
  const [threads, setThreads] = useState<ThreadResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [creatingThread, setCreatingThread] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // User info
  const [userEmail, setUserEmail] = useState<string | null>(null);

  // Home-screen input
  const [centralInput, setCentralInput] = useState('');
  const [showGuide, setShowGuide] = useState(false);

  // My Documents
  const [userDocs, setUserDocs] = useState<UserDocument[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [deletingDocIds, setDeletingDocIds] = useState<Set<string>>(new Set());

  // Pending (ingesting) docs + polling
  const [pendingDocs, setPendingDocs] = useState<PendingDoc[]>([]);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Public (system) filings
  const [publicDocs, setPublicDocs] = useState<PublicDoc[]>([]);

  // Home-screen upload state
  const homeFileInputRef = useRef<HTMLInputElement>(null);
  const [homeUploadState, setHomeUploadState] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [homeUploadName, setHomeUploadName] = useState<string | null>(null);
  const [homeUploadError, setHomeUploadError] = useState<string | null>(null);

  // ─── Polling: detect when background ingestion finishes ──────────────────

  useEffect(() => {
    const activePending = pendingDocs.filter(p => !p.timedOut);

    if (activePending.length === 0) {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      return;
    }

    if (pollingRef.current) return; // already running

    pollingRef.current = setInterval(async () => {
      try {
        const docs = await listUserDocuments();
        setUserDocs(docs);

        setPendingDocs(prev =>
          prev.map(pd => {
            if (pd.timedOut) return pd; // already in error state
            const found = docs.some(d => d.filename === pd.filename);
            if (found) return null as unknown as PendingDoc; // mark for removal
            const timedOut = Date.now() - pd.startTime > INGESTION_TIMEOUT_MS;
            return timedOut ? { ...pd, timedOut: true } : pd;
          }).filter(Boolean) as PendingDoc[]
        );
      } catch {
        // silent — will retry on next tick
      }
    }, POLLING_INTERVAL_MS);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [pendingDocs.filter(p => !p.timedOut).length]);

  // ─── Home screen upload ───────────────────────────────────────────────────

  const handleHomeFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setHomeUploadError('Only PDF files are supported.');
      setHomeUploadState('error');
      return;
    }
    setHomeUploadName(file.name);
    setHomeUploadState('uploading');
    setHomeUploadError(null);

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

      setHomeUploadState('success');
      setPendingDocs(prev => [...prev, { filename: file.name, startTime: Date.now() }]);
      setTimeout(() => { setHomeUploadState('idle'); setHomeUploadName(null); }, 4000);
    } catch (err) {
      setHomeUploadError(err instanceof Error ? err.message : 'Upload failed.');
      setHomeUploadState('error');
    } finally {
      if (homeFileInputRef.current) homeFileInputRef.current.value = '';
    }
  };

  // ─── Chat window upload callback ──────────────────────────────────────────

  const handleChatUploadStart = useCallback((filename: string) => {
    setPendingDocs(prev => [...prev, { filename, startTime: Date.now() }]);
  }, []);

  // ─── Thread list + user info ──────────────────────────────────────────────

  const loadDashboardData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const { data: { session } } = await supabase.auth.getSession();
      if (session?.user) {
        setUserEmail(session.user.email || 'Analyst');
      }

      const threadList = await listThreads();
      setThreads(threadList);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
      setError(err instanceof Error ? err.message : 'Failed to sync database.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadDashboardData(); }, [loadDashboardData]);

  // ─── Load private + public docs once on mount ─────────────────────────────

  useEffect(() => {
    async function loadDocs() {
      setDocsLoading(true);
      try {
        const [myDocs, pubDocs] = await Promise.all([
          listUserDocuments(),
          listPublicDocuments(),
        ]);
        setUserDocs(myDocs);
        setPublicDocs(pubDocs);
      } catch {
        // Silently fail — sidebar degrades gracefully
      } finally {
        setDocsLoading(false);
      }
    }
    loadDocs();
  }, []);

  // ─── Thread actions ───────────────────────────────────────────────────────

  const handleCreateThread = async () => {
    try {
      setCreatingThread(true);
      const title = `Session #${threads.length + 1}`;
      const newThread = await createThread(title);
      setThreads(prev => [newThread, ...prev]);
      navigate(`/thread/${newThread.id}`);
    } catch (err) {
      console.error('Failed to create thread:', err);
    } finally {
      setCreatingThread(false);
    }
  };

  const handleDeleteThread = async (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm('Are you sure you want to permanently delete this session?')) return;
    try {
      await deleteThread(id);
      setThreads(prev => prev.filter(t => t.id !== id));
      if (threadId === id) navigate('/');
    } catch (err) {
      console.error('Failed to delete thread:', err);
    }
  };

  // ─── Document delete (with "Deleting..." status) ─────────────────────────

  const handleDeleteDoc = async (docId: string, filename: string) => {
    if (!confirm(`Delete "${filename}" from your library?\n\nThis cannot be undone.`)) return;

    // Show "Deleting..." state immediately
    setDeletingDocIds(prev => new Set(prev).add(docId));

    try {
      await deleteDocument(docId);
      // Remove from list once confirmed deleted
      setUserDocs(prev => prev.filter(d => d.id !== docId));
    } catch (err) {
      console.error('Failed to delete document:', err);
    } finally {
      setDeletingDocIds(prev => {
        const next = new Set(prev);
        next.delete(docId);
        return next;
      });
    }
  };

  // ─── Home-screen new-thread submission ────────────────────────────────────

  const handleSuggestionClick = async (prompt: string) => {
    if (!prompt.trim()) return;
    try {
      setCreatingThread(true);
      const shortTitle = prompt.length > 25 ? `${prompt.substring(0, 25)}...` : prompt;
      const newThread = await createThread(shortTitle);
      setThreads(prev => [newThread, ...prev]);
      setCentralInput('');
      navigate(`/thread/${newThread.id}`, { state: { initialPrompt: prompt } });
    } catch (err) {
      console.error('Failed to execute suggestion:', err);
    } finally {
      setCreatingThread(false);
    }
  };

  const handleSignOut = async () => { await supabase.auth.signOut(); };

  const activeThread = threads.find(t => t.id === threadId);

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#030712] text-slate-100 font-sans relative">

      {/* ═══ SIDEBAR ═══════════════════════════════════════════════════════ */}
      <aside className="w-72 border-r border-slate-900 bg-[#060813] flex flex-col h-full shrink-0 select-none">

        {/* App Branding */}
        <div className="h-16 px-6 border-b border-slate-900 flex items-center">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-600 text-white font-bold text-sm shadow-md shadow-blue-500/20">
              α
            </div>
            <span className="text-xs font-bold tracking-wider text-slate-200">ALPHA COPILOT</span>
          </div>
        </div>

        {/* New Thread Button */}
        <div className="p-4">
          <button
            onClick={handleCreateThread}
            disabled={creatingThread}
            className="w-full bg-[#111827] hover:bg-[#1f2937] text-slate-200 border border-slate-800 rounded-xl py-3 px-4 font-semibold text-xs flex items-center justify-center gap-2 shadow-lg cursor-pointer disabled:opacity-50 group transition-all"
          >
            {creatingThread
              ? <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
              : <Plus className="h-4 w-4 text-blue-500 group-hover:rotate-90 transition-transform duration-200" />
            }
            <span>New Thread</span>
          </button>
        </div>

        {/* Scrollable sidebar body */}
        <div className="flex-1 overflow-y-auto px-3 space-y-5 py-2 scrollbar-none">

          {/* ── Active Sessions ─────────────────────────────────────────── */}
          <div className="space-y-0.5">
            {loading ? (
              <div className="flex items-center justify-center py-6">
                <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
              </div>
            ) : error ? (
              <div className="flex flex-col gap-1.5 rounded-xl bg-red-500/5 border border-red-500/10 p-3 mx-1">
                <div className="flex items-center gap-2 text-red-400">
                  <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                  <span className="text-[10px] font-medium leading-tight">Could not load threads</span>
                </div>
                <button
                  onClick={loadDashboardData}
                  className="text-[10px] font-semibold text-blue-400 hover:text-blue-300 underline text-left cursor-pointer transition-colors"
                >
                  Retry
                </button>
              </div>
            ) : threads.length === 0 ? (
              <p className="text-left py-4 text-slate-500 text-[11px] px-3 font-medium">No active sessions.</p>
            ) : (
              threads.map(t => {
                const isActive = t.id === threadId;
                return (
                  <div
                    key={t.id}
                    onClick={() => navigate(`/thread/${t.id}`)}
                    className={cn(
                      "group flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 cursor-pointer",
                      isActive
                        ? "bg-[#111827] text-blue-400 border border-slate-800"
                        : "bg-transparent text-slate-400 hover:bg-slate-900/40 hover:text-slate-200"
                    )}
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <MessageSquare className={cn("h-3.5 w-3.5 shrink-0", isActive ? "text-blue-500" : "text-slate-600 group-hover:text-slate-400")} />
                      <span className="truncate pr-1">{t.title}</span>
                    </div>
                    <button
                      onClick={e => handleDeleteThread(e, t.id)}
                      className="opacity-0 group-hover:opacity-100 p-0.5 rounded transition-all duration-200"
                      title="Delete session"
                    >
                      <Trash2 className="h-3.5 w-3.5 text-slate-600 hover:text-red-400 transition-colors" />
                    </button>
                  </div>
                );
              })
            )}
          </div>

          {/* ── My Documents ─────────────────────────────────────────────── */}
          <div className="space-y-1.5">
            <div className="px-2 py-1.5 text-[9px] font-bold text-slate-500 uppercase tracking-widest font-mono flex items-center gap-1.5 border-t border-slate-900 pt-3">
              <Database className="h-3 w-3" />
              <span>My Documents</span>
            </div>

            <div className="space-y-1 px-1">
              {docsLoading ? (
                <div className="flex items-center gap-2 px-2 py-2 text-slate-600 text-[11px]">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  <span>Loading...</span>
                </div>
              ) : (
                <>
                  {/* Actively ingesting docs */}
                  {pendingDocs.map(pd => (
                    <div
                      key={pd.filename}
                      className={cn(
                        "flex items-center justify-between px-2 py-1.5 rounded-lg border",
                        pd.timedOut
                          ? "bg-amber-950/15 border-amber-900/30"
                          : "bg-blue-950/15 border-blue-900/30"
                      )}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        {pd.timedOut
                          ? <AlertTriangle className="h-2.5 w-2.5 text-amber-400 shrink-0" />
                          : <Loader2 className="h-2.5 w-2.5 text-blue-400 animate-spin shrink-0" />
                        }
                        <span
                          className={cn("text-[11px] truncate max-w-[130px]", pd.timedOut ? "text-amber-400/70" : "text-blue-400/70")}
                          title={pd.filename}
                        >
                          {pd.filename}
                        </span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className={cn(
                          "text-[9px] font-mono px-1.5 py-0.5 rounded shrink-0",
                          pd.timedOut
                            ? "bg-amber-950/40 text-amber-400"
                            : "bg-blue-950/40 text-blue-400"
                        )}>
                          {pd.timedOut ? 'Slow...' : 'Ingesting'}
                        </span>
                        {pd.timedOut && (
                          <button
                            onClick={() => setPendingDocs(prev => prev.filter(p => p.filename !== pd.filename))}
                            className="p-0.5 text-slate-600 hover:text-slate-400 transition-colors"
                            title="Dismiss"
                          >
                            <X className="h-3 w-3" />
                          </button>
                        )}
                      </div>
                    </div>
                  ))}

                  {/* Uploaded private docs */}
                  {userDocs.length === 0 && pendingDocs.length === 0 ? (
                    <p className="text-[11px] text-slate-600 px-2 py-1 leading-snug">
                      No uploads yet. Use the 📎 in chat.
                    </p>
                  ) : (
                    userDocs.map(doc => {
                      const isDeleting = deletingDocIds.has(doc.id);
                      return (
                        <div
                          key={doc.id}
                          className={cn(
                            "group flex items-center justify-between px-2 py-1.5 rounded-lg border transition-all",
                            isDeleting
                              ? "bg-red-950/10 border-red-900/30"
                              : "bg-slate-900/10 border-slate-900/40 hover:border-slate-800/60"
                          )}
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            {isDeleting ? (
                              <Loader2 className="h-2.5 w-2.5 text-red-400 animate-spin shrink-0" />
                            ) : (
                              <div className="h-1.5 w-1.5 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)] shrink-0" />
                            )}
                            <span
                              className={cn("text-[11px] truncate max-w-[130px]", isDeleting ? "text-red-400/70" : "text-slate-400")}
                              title={doc.filename}
                            >
                              {doc.filename}
                            </span>
                          </div>
                          <div className="flex items-center gap-1 shrink-0">
                            {isDeleting ? (
                              <span className="text-[9px] font-mono bg-red-950/40 text-red-400 px-1.5 py-0.5 rounded">
                                Deleting
                              </span>
                            ) : (
                              <>
                                <span className="text-[9px] font-mono bg-slate-900 text-slate-500 px-1 py-0.5 rounded">PDF</span>
                                <button
                                  onClick={() => handleDeleteDoc(doc.id, doc.filename)}
                                  className="opacity-0 group-hover:opacity-100 p-0.5 rounded transition-all duration-200 ml-0.5"
                                  title={`Delete ${doc.filename}`}
                                >
                                  <Trash2 className="h-3 w-3 text-slate-600 hover:text-red-400 transition-colors" />
                                </button>
                              </>
                            )}
                          </div>
                        </div>
                      );
                    })
                  )}
                </>
              )}
            </div>
          </div>

          {/* ── Pre-loaded Filings (Public / System) ─────────────────────── */}
          {publicDocs.length > 0 && (
            <div className="space-y-1.5">
              <div className="px-2 py-1.5 text-[9px] font-bold text-slate-500 uppercase tracking-widest font-mono flex items-center gap-1.5 border-t border-slate-900 pt-3">
                <Lock className="h-3 w-3" />
                <span>Pre-loaded Filings</span>
              </div>
              <div className="space-y-1 px-1">
                {publicDocs.map(doc => (
                  <div
                    key={doc.id}
                    className="flex items-center justify-between px-2 py-1.5 rounded-lg bg-slate-900/10 border border-slate-900/40"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <div className="h-1.5 w-1.5 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)] shrink-0" />
                      <span className="text-[11px] text-slate-400 truncate max-w-[135px]" title={doc.filename}>
                        {doc.filename}
                      </span>
                    </div>
                    <span className="text-[9px] font-mono bg-slate-900 text-slate-500 px-1.5 py-0.5 rounded shrink-0 flex items-center gap-1">
                      <Lock className="h-2 w-2 text-slate-650" />
                      {doc.ticker}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* User Profile Footer */}
        <div className="p-4 border-t border-slate-900 bg-[#060813] flex flex-col gap-3">
          <div className="flex items-center gap-3 px-1">
            <div className="h-8 w-8 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-450">
              <User className="h-4 w-4" />
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-xs text-slate-350 truncate font-semibold">{userEmail}</span>
            </div>
          </div>
          <Button
            onClick={handleSignOut}
            variant="outline"
            className="w-full border-slate-900 bg-slate-950 hover:bg-slate-900/60 text-slate-400 hover:text-slate-200 rounded-xl py-4 flex items-center justify-center gap-2 text-xs font-semibold cursor-pointer shadow-sm"
          >
            <LogOut className="h-4 w-4" />
            <span>Sign Out</span>
          </Button>
        </div>
      </aside>

      {/* ═══ MAIN CONTENT ═══════════════════════════════════════════════════ */}
      <main className="flex-1 h-full overflow-hidden bg-[#030712] relative flex flex-col">
        {activeThread ? (
          <ChatWindow
            key={activeThread.id}
            threadId={activeThread.id}
            threadTitle={activeThread.title}
            onUploadStart={handleChatUploadStart}
          />
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center relative px-6">
            {/* Horizon glow */}
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[140%] h-[350px] bg-gradient-to-t from-blue-750/10 via-transparent to-transparent rounded-[100%] blur-3xl pointer-events-none translate-y-48" />

            <div className="w-full max-w-xl text-center space-y-6 relative z-10">
              <div className="space-y-1">
                <h1 className="text-3xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-b from-slate-50 to-slate-300">
                  Hello, Analyst.
                </h1>
                <p className="text-xl font-light text-slate-400">How can I help you today?</p>
              </div>

              {/* Upload banner */}
              {homeUploadState !== 'idle' && (
                <div className={cn(
                  "flex items-center justify-between gap-3 rounded-xl px-3 py-2 text-xs font-semibold border transition-all text-left",
                  homeUploadState === 'uploading' && "bg-blue-950/20 border-blue-500/20 text-blue-300",
                  homeUploadState === 'success'  && "bg-green-950/20 border-green-500/20 text-green-300",
                  homeUploadState === 'error'    && "bg-red-950/20 border-red-500/20 text-red-300",
                )}>
                  <div className="flex items-center gap-2 min-w-0">
                    {homeUploadState === 'uploading' && <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />}
                    {homeUploadState === 'success'  && <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />}
                    {homeUploadState === 'error'    && <AlertCircle className="h-3.5 w-3.5 shrink-0" />}
                    <span className="truncate">
                      {homeUploadState === 'uploading' && `Uploading "${homeUploadName}"...`}
                      {homeUploadState === 'success'  && `"${homeUploadName}" received — ingesting in background.`}
                      {homeUploadState === 'error'    && (homeUploadError || 'Upload failed.')}
                    </span>
                  </div>
                  <button type="button" onClick={() => { setHomeUploadState('idle'); setHomeUploadName(null); setHomeUploadError(null); }} className="shrink-0 opacity-60 hover:opacity-100">
                    <X className="h-3 w-3" />
                  </button>
                </div>
              )}

              {/* Hidden PDF file input */}
              <input ref={homeFileInputRef} type="file" accept=".pdf,application/pdf" className="hidden" onChange={handleHomeFileUpload} />

              {/* Central Input Form */}
              <form
                onSubmit={e => { e.preventDefault(); handleSuggestionClick(centralInput); }}
                className="bg-[#070b16]/80 border border-slate-800/80 rounded-2xl p-4 shadow-2xl focus-within:border-blue-500/50 focus-within:ring-2 focus-within:ring-blue-500/10 transition-all duration-300 text-left"
              >
                <textarea
                  value={centralInput}
                  onChange={e => setCentralInput(e.target.value)}
                  placeholder="Ask anything..."
                  className="w-full bg-transparent border-0 outline-none resize-none text-slate-200 placeholder-slate-550 text-sm h-20 focus:ring-0"
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); } }}
                />
                <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-900/80">
                  <div className="flex items-center gap-2">
                    <span className="rounded-xl px-4 py-1.5 text-[11px] font-semibold flex items-center gap-1.5 bg-blue-600 text-slate-50 shadow-md shadow-blue-500/10">
                      <Search className="h-3 w-3" />
                      <span>Search Filings</span>
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-slate-500">
                    <button
                      type="button"
                      onClick={e => { e.preventDefault(); e.stopPropagation(); homeFileInputRef.current?.click(); }}
                      disabled={homeUploadState === 'uploading'}
                      className="hover:text-blue-400 transition-colors p-1 cursor-pointer disabled:opacity-40"
                      title="Upload PDF document"
                    >
                      {homeUploadState === 'uploading' ? <Loader2 className="h-4 w-4 animate-spin text-blue-400" /> : <Paperclip className="h-4 w-4" />}
                    </button>
                    <button
                      type="submit"
                      disabled={!centralInput.trim() || creatingThread}
                      className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white rounded-lg p-1.5 shadow transition-colors cursor-pointer"
                    >
                      <ArrowRight className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </div>
        )}
      </main>

      {/* ═══ HELP BUTTON ════════════════════════════════════════════════════ */}
      <div className="absolute bottom-6 right-6 z-30">
        <button
          onClick={() => setShowGuide(true)}
          className="bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20 h-11 w-11 rounded-full flex items-center justify-center cursor-pointer transition-transform hover:scale-105"
          title="Workspace Help & Documentation"
        >
          <HelpCircle className="h-5 w-5" />
        </button>
      </div>

      {/* ═══ GUIDE MODAL ════════════════════════════════════════════════════ */}
      {showGuide && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-fade-in">
          <div className="bg-[#0b0f19] border border-slate-800 rounded-2xl w-full max-w-md p-6 text-slate-200 shadow-2xl relative space-y-4">
            <h3 className="text-base font-bold text-slate-100 flex items-center gap-2 border-b border-slate-900 pb-2.5">
              <HelpCircle className="h-4 w-4 text-blue-500" />
              Alpha CoPilot Analyst Guide
            </h3>
            <div className="space-y-4 text-xs text-slate-400 leading-relaxed overflow-y-auto max-h-[320px] pr-1 scrollbar-thin">
              <div>
                <h4 className="font-bold text-slate-200 mb-0.5">What is Alpha Copilot?</h4>
                <p>An AI-powered document intelligence workspace for financial analysts to interrogate annual reports with strict factuality and citation audits.</p>
              </div>
              <div>
                <h4 className="font-bold text-slate-200 mb-0.5">Pre-loaded Filings vs. My Documents</h4>
                <p>The sidebar shows two document sections. <strong className="text-slate-300">Pre-loaded Filings</strong> are system documents (TCS, Reliance, etc.) always available in retrieval — they cannot be deleted. <strong className="text-slate-300">My Documents</strong> are PDFs you personally uploaded and can delete anytime.</p>
              </div>
              <div>
                <h4 className="font-bold text-slate-200 mb-0.5">Hybrid Dense-Sparse Search</h4>
                <p>For each query, the system combines dense vector search (pgvector semantic match) and sparse keyword search, merged via Reciprocal Rank Fusion (RRF) to retrieve the most exact source excerpts.</p>
              </div>
              <div>
                <h4 className="font-bold text-slate-200 mb-0.5">Document Upload & Ingestion</h4>
                <p>PDFs are ingested in the background — large documents can take several minutes. The sidebar shows "Ingesting" while processing. If it takes longer than expected, it will show "Slow..." but will still complete.</p>
              </div>
              <div>
                <h4 className="font-bold text-slate-200 mb-0.5">Speculation Policy Guard</h4>
                <p>Alpha strictly enforces a grounding validation policy. The system will refuse to answer or compute metrics not explicitly found in the retrieved filing segments.</p>
              </div>
            </div>
            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setShowGuide(false)}
                className="bg-blue-600 hover:bg-blue-500 text-slate-100 font-semibold px-4 py-2 rounded-xl text-xs transition-colors cursor-pointer"
              >
                Dismiss Guide
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
