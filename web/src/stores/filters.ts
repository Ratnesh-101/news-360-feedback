import { create } from 'zustand';
import type { Sentiment, LangCode, ChatFilters } from '@/types/domain';
import type { ArticleFilters } from '@/types/api';

interface FilterState {
  q: string;
  from?: string;
  to?: string;
  ministries: string[];
  languages: LangCode[];
  categories: string[];
  sentiments: Sentiment[];
  setQ: (q: string) => void;
  setDateRange: (from?: string, to?: string) => void;
  toggleMinistry: (m: string) => void;
  toggleLanguage: (l: LangCode) => void;
  toggleCategory: (c: string) => void;
  toggleSentiment: (s: Sentiment) => void;
  resetFilters: () => void;
  getApiParams: () => ArticleFilters;
  getChatFilters: () => ChatFilters;
}

export const useFilters = create<FilterState>((set, get) => ({
  q: '',
  ministries: [],
  languages: [],
  categories: [],
  sentiments: [],

  setQ: (q) => set({ q }),
  setDateRange: (from, to) => set({ from, to }),

  toggleMinistry: (m) =>
    set((state) => ({
      ministries: state.ministries.includes(m)
        ? state.ministries.filter((x) => x !== m)
        : [...state.ministries, m],
    })),

  toggleLanguage: (l) =>
    set((state) => ({
      languages: state.languages.includes(l)
        ? state.languages.filter((x) => x !== l)
        : [...state.languages, l],
    })),

  toggleCategory: (c) =>
    set((state) => ({
      categories: state.categories.includes(c)
        ? state.categories.filter((x) => x !== c)
        : [...state.categories, c],
    })),

  toggleSentiment: (s) =>
    set((state) => ({
      sentiments: state.sentiments.includes(s)
        ? state.sentiments.filter((x) => x !== s)
        : [...state.sentiments, s],
    })),

  resetFilters: () =>
    set({
      q: '',
      from: undefined,
      to: undefined,
      ministries: [],
      languages: [],
      categories: [],
      sentiments: [],
    }),

  getApiParams: () => {
    const s = get();
    return {
      q: s.q || undefined,
      from: s.from,
      to: s.to,
      ministry: s.ministries.length ? s.ministries : undefined,
      language: s.languages.length ? s.languages : undefined,
      category: s.categories.length ? s.categories : undefined,
      sentiment: s.sentiments.length ? s.sentiments : undefined,
    };
  },

  getChatFilters: () => {
    const s = get();
    return {
      ministries: s.ministries.length ? s.ministries : undefined,
      from: s.from,
      to: s.to,
      languages: s.languages.length ? s.languages : undefined,
    };
  },
}));
