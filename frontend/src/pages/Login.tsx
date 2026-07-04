import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { supabase } from '@/lib/supabase';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '@/components/ui/card';
import { Lock, Mail, AlertCircle, Eye, EyeOff, Loader2 } from 'lucide-react';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const { error: authError } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (authError) {
        throw authError;
      }
      
      navigate('/');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to sign in. Please check your credentials.';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center bg-[#030712] px-4 py-12 sm:px-6 lg:px-8 overflow-hidden font-sans">
      {/* Curved atmosphere planet glow at bottom */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[140%] h-[350px] bg-gradient-to-t from-blue-700/10 via-transparent to-transparent rounded-[100%] blur-3xl pointer-events-none translate-y-48"></div>

      {/* Ambient background highlight */}
      <div className="absolute top-1/4 left-1/4 h-96 w-96 rounded-full bg-blue-600/5 blur-[120px] pointer-events-none"></div>

      <Card className="w-full max-w-md border-slate-800 bg-[#070b16]/65 backdrop-blur-xl shadow-2xl text-slate-100 relative z-10 rounded-2xl">
        <CardHeader className="space-y-1.5 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-blue-950/20 border border-blue-500/25 text-blue-400 shadow-md">
            <Lock className="h-5 w-5" />
          </div>
          <CardTitle className="text-2xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-b from-slate-50 to-slate-300">
            Welcome Back
          </CardTitle>
          <CardDescription className="text-slate-400 text-xs font-medium">
            Sign in to access your financial document copilot
          </CardDescription>
        </CardHeader>
        
        <form onSubmit={handleLogin}>
          <CardContent className="space-y-4">
            {error && (
              <div className="flex items-center gap-2 rounded-xl bg-red-500/5 border border-red-500/15 p-3 text-xs font-semibold text-red-400">
                <AlertCircle className="h-4 w-4 shrink-0 text-red-500" />
                <span>{error}</span>
              </div>
            )}
            
            <div className="space-y-1.5">
              <Label htmlFor="email" className="text-slate-300 text-[10px] font-bold uppercase tracking-wider font-mono">
                Email Address
              </Label>
              <div className="relative">
                <Mail className="absolute top-1/2 left-3.5 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <Input
                  id="email"
                  type="email"
                  placeholder="name@company.com"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={loading}
                  className="pl-10 bg-[#040814]/50 border-slate-850 focus:border-blue-500 focus:ring-blue-500/10 text-slate-200 placeholder-slate-600 rounded-xl"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="password" className="text-slate-300 text-[10px] font-bold uppercase tracking-wider font-mono">
                Password
              </Label>
              <div className="relative">
                <Lock className="absolute top-1/2 left-3.5 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <Input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="••••••••"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={loading}
                  className="pl-10 pr-10 bg-[#040814]/50 border-slate-850 focus:border-blue-500 focus:ring-blue-500/10 text-slate-200 placeholder-slate-600 rounded-xl"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute top-1/2 right-3.5 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>
          </CardContent>

          <CardFooter className="flex flex-col gap-4 bg-transparent border-t-0">
            <Button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2.5 rounded-xl shadow-lg shadow-blue-500/10 hover:shadow-blue-500/15 transition-all cursor-pointer"
            >
              {loading ? (
                <div className="flex items-center justify-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Signing in...</span>
                </div>
              ) : (
                'Sign In'
              )}
            </Button>
            
            <div className="text-center text-xs font-semibold text-slate-500">
              Don't have an account?{' '}
              <Link to="/signup" className="text-blue-400 hover:text-blue-350 font-semibold hover:underline transition-colors">
                Create one
              </Link>
            </div>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
