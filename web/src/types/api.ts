import type { ISODate, Sentiment, ChatSource } from './domain';

export interface Paginated<T> {
  items: T[];
  nextCursor: string | null;
  total: number;
}

export interface ArticleFilters {
  q?: string;
  from?: ISODate;
  to?: ISODate;
  ministry?: string[];
  language?: string[];
  category?: string[];
  sentiment?: Sentiment[];
  sort?: 'published_desc' | 'score_desc';
  offset?: number;
  limit?: number;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  details?: unknown;
}

export type ChatStreamEvent =
  | { type: 'sources'; sources: ChatSource[] }
  | { type: 'token'; text: string }
  | { type: 'done'; messageId: string }
  | { type: 'error'; message: string };
