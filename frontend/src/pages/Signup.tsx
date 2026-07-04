import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '@/components/ui/card';
import { ShieldAlert, ArrowLeft } from 'lucide-react';

export default function Signup() {
  return (
    <div className="relative flex min-h-screen items-center justify-center bg-[#030712] px-4 py-12 sm:px-6 lg:px-8 overflow-hidden font-sans">
      {/* Curved atmosphere planet glow at bottom */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[140%] h-[350px] bg-gradient-to-t from-blue-700/10 via-transparent to-transparent rounded-[100%] blur-3xl pointer-events-none translate-y-48"></div>

      {/* Ambient background highlight */}
      <div className="absolute top-1/4 left-1/4 h-96 w-96 rounded-full bg-blue-600/5 blur-[120px] pointer-events-none"></div>

      <Card className="w-full max-w-md border-slate-800 bg-[#070b16]/65 backdrop-blur-xl shadow-2xl text-slate-100 relative z-10 rounded-2xl">
        <CardHeader className="space-y-1.5 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-amber-950/20 border border-amber-500/25 text-amber-400 shadow-md">
            <ShieldAlert className="h-5 w-5" />
          </div>
          <CardTitle className="text-2xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-b from-slate-50 to-slate-300">
            Signups Disabled
          </CardTitle>
          <CardDescription className="text-slate-400 text-xs font-medium">
            Registrations are restricted for this demo
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4 text-center text-sm text-slate-300 leading-relaxed">
          <p>
            To prevent rate-limit abuse and conserve API usage, new account registrations are currently disabled for this public preview.
          </p>
          <p className="text-xs text-slate-400 bg-[#040814]/40 border border-slate-800/40 p-3 rounded-xl">
            Please use the <strong>Demo Guest Access</strong> credentials provided directly on the sign-in page to explore the application.
          </p>
        </CardContent>

        <CardFooter className="flex flex-col gap-4 bg-transparent border-t-0">
          <Link to="/login" className="w-full">
            <Button
              className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2.5 rounded-xl shadow-lg shadow-blue-500/10 hover:shadow-blue-500/15 transition-all cursor-pointer flex items-center justify-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              <span>Back to Sign In</span>
            </Button>
          </Link>
        </CardFooter>
      </Card>
    </div>
  );
}
