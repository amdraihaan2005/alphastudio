import { useEffect, useState } from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { supabase } from '@/lib/supabase';
import type { Session } from '@supabase/supabase-js';
import { Loader2 } from 'lucide-react';

export default function ProtectedRoute() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setLoading(false);
    });

    // Listen for auth state changes (sign in, sign out, token refresh)
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      setLoading(false);
    });

    return () => subscription.unsubscribe();
  }, []);

  if (loading) {
    return (
      <div className="relative flex min-h-screen items-center justify-center bg-[#030712] overflow-hidden">
        {/* Curved atmosphere planet glow at bottom */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[140%] h-[350px] bg-gradient-to-t from-blue-700/10 via-transparent to-transparent rounded-[100%] blur-3xl pointer-events-none translate-y-48" />

        {/* Ambient background highlight */}
        <div className="absolute top-1/4 left-1/4 h-96 w-96 rounded-full bg-blue-600/5 blur-[120px] pointer-events-none" />

        <div className="relative z-10 flex flex-col items-center gap-3">
          <Loader2 className="h-7 w-7 animate-spin text-blue-400" />
          <span className="text-sm font-medium text-slate-400 tracking-wide">Verifying session...</span>
        </div>
      </div>
    );
  }

  if (!session) {
    return <Navigate to="/login" replace />;
  }

  // Session is active, expose user context
  return <Outlet context={{ session }} />;
}
