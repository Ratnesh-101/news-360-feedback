/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ink: 'var(--ink)',
        navy: 'var(--navy)',
        chakra: 'var(--chakra)',
        paper: 'var(--paper)',
        surface: 'var(--surface)',
        rule: 'var(--rule)',
        pos: 'var(--pos)',
        neg: 'var(--neg)',
        warn: 'var(--warn)',
        neu: 'var(--neu)',
      },
      borderRadius: {
        panel: '10px',
        control: '6px',
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', '"Noto Sans Devanagari"', '"Noto Sans Gurmukhi"', 'system-ui', 'sans-serif'],
        serif: ['"Source Serif 4"', 'Georgia', 'serif'],
      },
    },
  },
  plugins: [],
}
