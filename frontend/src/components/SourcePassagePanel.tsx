import { useEffect, useState } from 'react';
import { getChunkDetails, type ChunkDetailsResponse } from '@/lib/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { 
  X, 
  Loader2, 
  AlertCircle, 
  BookOpen, 
  Building2, 
  Calendar, 
  ChevronDown, 
  ChevronUp,
  FileText
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface SourcePassagePanelProps {
  chunkId: string;
  onClose: () => void;
}

export default function SourcePassagePanel({ chunkId, onClose }: SourcePassagePanelProps) {
  const [chunk, setChunk] = useState<ChunkDetailsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showPreceding, setShowPreceding] = useState(false);
  const [showSucceeding, setShowSucceeding] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function fetchDetails() {
      try {
        setLoading(true);
        setError(null);
        // Reset accordion expansions on chunk change
        setShowPreceding(false);
        setShowSucceeding(false);

        const data = await getChunkDetails(chunkId);
        if (isMounted) {
          setChunk(data);
        }
      } catch (err) {
        console.error('Error fetching chunk details:', err);
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to retrieve chunk context.');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    fetchDetails();

    return () => {
      isMounted = false;
    };
  }, [chunkId]);

  return (
    <div className="w-96 md:w-[420px] border-l border-slate-900 bg-[#060813]/95 backdrop-blur-md flex flex-col h-full shrink-0 select-none animate-slide-in relative z-30">
      {/* Header */}
      <header className="h-16 px-6 border-b border-slate-900 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400">
            <BookOpen className="h-4 w-4" />
          </div>
          <div>
            <span className="text-xs font-bold tracking-wider text-slate-200">SOURCE AUDIT</span>
            <p className="text-[10px] text-slate-500 font-mono tracking-tight font-medium mt-0.5 leading-none">
              VERIFIABLE EVIDENCE BLOCK
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-slate-500 hover:text-slate-300 p-1.5 hover:bg-slate-900/50 rounded-lg transition-all cursor-pointer"
        >
          <X className="h-4 w-4" />
        </button>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-none select-text">
        {loading ? (
          <div className="flex flex-col items-center justify-center h-64 gap-3 text-slate-400">
            <Loader2 className="h-7 w-7 animate-spin text-blue-400" />
            <p className="text-xs font-medium font-mono">Retrieving source excerpt...</p>
          </div>
        ) : error ? (
          <div className="rounded-xl bg-red-500/5 border border-red-500/10 p-4 text-center">
            <AlertCircle className="h-7 w-7 text-red-500/60 mx-auto mb-2" />
            <h4 className="text-xs font-semibold text-slate-200">Context load failed</h4>
            <p className="text-[10px] text-slate-500 mt-1">{error}</p>
          </div>
        ) : chunk ? (
          <>
            {/* Metadata Badges */}
            <div className="grid grid-cols-2 gap-2 bg-slate-900/35 border border-slate-800/60 p-3.5 rounded-xl">
              <div className="flex items-center gap-2 text-slate-400">
                <Building2 className="h-3.5 w-3.5 text-slate-500" />
                <div className="flex flex-col">
                  <span className="text-[8px] font-bold text-slate-600 uppercase font-mono tracking-wider">Company</span>
                  <span className="text-xs font-semibold text-slate-300 truncate">{chunk.document.ticker}</span>
                </div>
              </div>
              <div className="flex items-center gap-2 text-slate-400">
                <Calendar className="h-3.5 w-3.5 text-slate-500" />
                <div className="flex flex-col">
                  <span className="text-[8px] font-bold text-slate-600 uppercase font-mono tracking-wider">Filing Year</span>
                  <span className="text-xs font-semibold text-slate-300">FY{chunk.document.year}</span>
                </div>
              </div>
              <div className="col-span-2 border-t border-slate-800/60 my-1"></div>
              <div className="col-span-2 flex items-center gap-2 text-slate-400">
                <FileText className="h-3.5 w-3.5 text-slate-500" />
                <div className="flex flex-col min-w-0">
                  <span className="text-[8px] font-bold text-slate-600 uppercase font-mono tracking-wider">Filing / File</span>
                  <span className="text-[11px] font-semibold text-slate-300 truncate font-mono">{chunk.document.filename}</span>
                </div>
              </div>
            </div>

            {/* Section & Page Info */}
            <div className="space-y-1">
              <span className="text-[9px] font-bold text-blue-400 uppercase tracking-widest font-mono">Location Context</span>
              <h3 className="text-sm font-bold text-slate-200">
                {chunk.section_name || 'General Content'}
              </h3>
              <p className="text-xs text-slate-500 font-medium">Page {chunk.page_number}</p>
            </div>

            {/* Excerpt Container */}
            <div className="space-y-4">
              {/* Preceding context accordion */}
              {chunk.preceding_text && (
                <div className="border border-slate-800/50 rounded-xl overflow-hidden bg-slate-900/10">
                  <button
                    onClick={() => setShowPreceding(!showPreceding)}
                    className="w-full px-4 py-2.5 flex items-center justify-between text-slate-500 hover:text-slate-400 text-[10px] font-bold uppercase font-mono transition-colors cursor-pointer"
                  >
                    <span>Preceding Text</span>
                    {showPreceding ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                  </button>
                  {showPreceding && (
                    <div className="px-4 pb-3 pt-2 text-xs leading-relaxed font-sans border-t border-slate-800/50 bg-slate-900/5 select-text overflow-hidden">
                      <CitationMarkdown text={chunk.preceding_text} />
                    </div>
                  )}
                </div>
              )}

              {/* Main chunk text */}
              <div className="relative rounded-xl border border-blue-500/10 bg-gradient-to-b from-blue-950/5 to-indigo-950/5 p-5 shadow-inner">
                <div className="absolute top-2.5 right-3 px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/20 text-blue-400 font-mono text-[9px] font-bold tracking-wider select-none">
                  CITED TEXT
                </div>
                <div className="text-sm leading-relaxed font-sans select-text pt-2 overflow-hidden">
                  <CitationMarkdown text={chunk.text_content} />
                </div>
              </div>

              {/* Succeeding context accordion */}
              {chunk.succeeding_text && (
                <div className="border border-slate-800/50 rounded-xl overflow-hidden bg-slate-900/10">
                  <button
                    onClick={() => setShowSucceeding(!showSucceeding)}
                    className="w-full px-4 py-2.5 flex items-center justify-between text-slate-500 hover:text-slate-400 text-[10px] font-bold uppercase font-mono transition-colors cursor-pointer"
                  >
                    <span>Succeeding Text</span>
                    {showSucceeding ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                  </button>
                  {showSucceeding && (
                    <div className="px-4 pb-3 pt-2 text-xs leading-relaxed font-sans border-t border-slate-800/50 bg-slate-900/5 select-text overflow-hidden">
                      <CitationMarkdown text={chunk.succeeding_text} />
                    </div>
                  )}
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="text-center text-slate-600 text-xs py-12 font-mono">
            Audit context unavailable.
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Local Citation Markdown Component ───────────────────────────────────────

function CitationMarkdown({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => (
          <p className="mb-2 last:mb-0 text-slate-350 font-sans leading-relaxed">{children}</p>
        ),
        strong: ({ children }) => (
          <strong className="font-semibold text-slate-200">{children}</strong>
        ),
        em: ({ children }) => (
          <em className="italic text-slate-300">{children}</em>
        ),
        ul: ({ children }) => (
          <ul className="list-disc list-outside ml-4 space-y-0.5 mb-2 text-slate-300">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal list-outside ml-4 space-y-0.5 mb-2 text-slate-300">{children}</ol>
        ),
        li: ({ children }) => (
          <li className="leading-relaxed">{children}</li>
        ),
        // Code
        code: ({ className, children }) => {
          const isBlock = Boolean(className);
          if (isBlock) {
            return (
              <pre className="bg-slate-900/80 border border-slate-850 rounded-lg p-2.5 overflow-x-auto mb-2">
                <code className={cn("font-mono text-xxs text-blue-300", className)}>
                  {children}
                </code>
              </pre>
            );
          }
          return (
            <code className="font-mono text-xxs bg-slate-900 text-blue-300 px-1 py-0.5 rounded">
              {children}
            </code>
          );
        },
        pre: ({ children }) => <>{children}</>,
        // Tables
        table: ({ children }) => (
          <div className="overflow-x-auto my-2 border border-slate-850 rounded-lg bg-slate-950/20">
            <table className="w-full text-xxs border-collapse text-slate-300 font-sans">{children}</table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-slate-900/40 text-slate-200 font-semibold border-b border-slate-850">{children}</thead>
        ),
        tbody: ({ children }) => <tbody className="divide-y divide-slate-850/40">{children}</tbody>,
        tr: ({ children }) => <tr className="hover:bg-slate-900/20 transition-colors">{children}</tr>,
        th: ({ children }) => <th className="px-2 py-1.5 text-left border-r border-slate-850/30 last:border-r-0 font-medium">{children}</th>,
        td: ({ children }) => <td className="px-2 py-1.5 border-r border-slate-850/20 last:border-r-0 font-mono text-[10px] leading-relaxed">{children}</td>,
      }}
    >
      {text}
    </ReactMarkdown>
  );
}
