import { create } from 'zustand';
import type { Article } from '@/types/domain';

interface UiState {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  chatOpen: boolean;
  setChatOpen: (open: boolean) => void;
  selectedArticle: Article | null;
  setSelectedArticle: (article: Article | null) => void;
}

export const useUiStore = create<UiState>((set) => {
  const initialTheme =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';

  return {
    theme: initialTheme,
    toggleTheme: () =>
      set((state) => {
        const next = state.theme === 'light' ? 'dark' : 'light';
        if (typeof document !== 'undefined') {
          if (next === 'dark') {
            document.documentElement.classList.add('dark');
          } else {
            document.documentElement.classList.remove('dark');
          }
        }
        return { theme: next };
      }),
    chatOpen: false,
    setChatOpen: (open) => set({ chatOpen: open }),
    selectedArticle: null,
    setSelectedArticle: (article) => set({ selectedArticle: article }),
  };
});
