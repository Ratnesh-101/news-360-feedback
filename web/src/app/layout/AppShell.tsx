import { useEffect } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  LayoutDashboard,
  Building2,
  Mic,
  FileText,
  MessageSquare,
  Sun,
  Moon,
  Filter,
  X,
} from 'lucide-react';
import { useUiStore } from '@/stores/ui';
import { useFilters } from '@/stores/filters';
import { ArticleDrawer } from '@/components/ArticleDrawer';
import { ChatDrawer } from '@/features/chat/ChatDrawer';

const NAV_ITEMS = [
  { to: '/', label: 'Command Center', icon: LayoutDashboard },
  { to: '/ministries', label: 'Ministry Radar', icon: Building2 },
  { to: '/audio', label: 'Audio Studio', icon: Mic },
  { to: '/briefing', label: 'Daily Briefing', icon: FileText },
];

export function AppShell() {
  const { theme, toggleTheme, chatOpen, setChatOpen } = useUiStore();
  const {
    languages,
    toggleLanguage,
    categories,
    toggleCategory,
    resetFilters,
  } = useFilters();

  // Listen for keyboard shortcut ⌘J or Ctrl+J to toggle chat
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'j') {
        e.preventDefault();
        setChatOpen(!chatOpen);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [chatOpen, setChatOpen]);

  const hasActiveFilters = languages.length > 0 || categories.length > 0;

  return (
    <div className="min-h-screen flex flex-col bg-paper text-ink transition-colors duration-200">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-40 border-b border-rule bg-surface/90 backdrop-blur-md print:hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between h-14">
          {/* Logo & Identity */}
          <div className="flex items-center gap-3">
            <div className="size-8 rounded-control bg-chakra text-white flex items-center justify-center font-serif font-bold text-lg shadow-sm">
              360°
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-serif font-bold text-sm tracking-tight text-ink">
                  News Feedback Intelligence
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-chakra/10 text-chakra font-semibold">
                  SIH1329
                </span>
              </div>
              <span className="text-[10px] font-mono text-ink/50 block -mt-0.5">
                Government of India Situation Room
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  className={({ isActive }) =>
                    `flex items-center gap-1.5 px-3 py-1.5 rounded-control text-xs font-semibold transition-colors ${
                      isActive
                        ? 'bg-chakra text-white shadow-sm'
                        : 'text-ink/70 hover:text-ink hover:bg-rule/30'
                    }`
                  }
                >
                  <Icon className="size-3.5" />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </nav>

          {/* Right Action Tools */}
          <div className="flex items-center gap-2">
            {/* Quick Lang toggles */}
            <div className="hidden lg:flex items-center gap-1 border-r border-rule pr-2 mr-1">
              {(['en', 'hi', 'mr'] as const).map((code) => {
                const isSelected = languages.includes(code);
                return (
                  <button
                    key={code}
                    onClick={() => toggleLanguage(code)}
                    className={`px-2 py-0.5 rounded text-[11px] font-mono uppercase font-bold transition-colors ${
                      isSelected
                        ? 'bg-chakra text-white'
                        : 'bg-rule/30 text-ink/60 hover:text-ink'
                    }`}
                  >
                    {code}
                  </button>
                );
              })}
            </div>

            {/* Clear filters badge if active */}
            {hasActiveFilters && (
              <button
                onClick={resetFilters}
                className="hidden sm:inline-flex items-center gap-1 text-[11px] font-mono text-neg hover:underline px-2 py-1 bg-neg/10 rounded"
              >
                <X className="size-3" /> Clear Filters
              </button>
            )}

            {/* RAG Chat Launcher */}
            <button
              onClick={() => setChatOpen(true)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-control bg-chakra/10 text-chakra hover:bg-chakra hover:text-white font-semibold text-xs transition-colors"
            >
              <MessageSquare className="size-3.5" />
              <span>Ask News</span>
              <kbd className="hidden sm:inline-block ml-1 px-1 rounded bg-surface/50 border border-rule/50 text-[10px] font-mono opacity-80">
                ⌘J
              </kbd>
            </button>

            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-control text-ink/60 hover:text-ink hover:bg-rule/30 transition-colors"
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation bar */}
        <div className="md:hidden flex items-center justify-around border-t border-rule/50 py-1.5 bg-paper/80">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `flex flex-col items-center gap-0.5 text-[10px] font-medium py-1 px-2 rounded ${
                    isActive ? 'text-chakra font-bold' : 'text-ink/60'
                  }`
                }
              >
                <Icon className="size-4" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 pb-16">
        <Outlet />
      </main>

      {/* Slide-over Drawers */}
      <ArticleDrawer />
      <ChatDrawer />
    </div>
  );
}
