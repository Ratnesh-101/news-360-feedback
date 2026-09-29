import type { LangCode } from '@/types/domain';

const LANG_LABELS: Record<string, { label: string; script: string }> = {
  hi: { label: 'Hindi', script: 'हिन्दी' },
  mr: { label: 'Marathi', script: 'मराठी' },
  en: { label: 'English', script: 'EN' },
  pa: { label: 'Punjabi', script: 'ਪੰਜਾਬੀ' },
  bn: { label: 'Bengali', script: 'বাংলা' },
  te: { label: 'Telugu', script: 'తెలుగు' },
  ta: { label: 'Tamil', script: 'தமிழ்' },
  gu: { label: 'Gujarati', script: 'ગુજરાતી' },
  kn: { label: 'Kannada', script: 'ಕನ್ನಡ' },
  ur: { label: 'Urdu', script: 'اردو' },
};

export function LangTag({ language }: { language: LangCode }) {
  const code = (language || 'en').toLowerCase();
  const info = LANG_LABELS[code] || { label: code.toUpperCase(), script: code.toUpperCase() };

  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-mono font-medium bg-rule/30 text-ink/80 border border-rule/50"
      title={`${info.label} (${info.script})`}
    >
      <span className="uppercase text-[10px] text-chakra font-semibold">{code}</span>
      <span className="text-ink/60 font-sans">{info.label}</span>
    </span>
  );
}
