import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#F7F9FC',
        surface: '#FFFFFF',
        'surface-elevated': '#FFFFFF',
        'surface-highlight': '#E7F4FC',
        sb: {
          cyan: '#12B8C2',
          navy: '#14264F',
          bg: '#F7F9FC',
          card: '#FFFFFF',
          border: '#DCE5EF',
          muted: '#71809A',
          active: '#E7F4FC',
        },
        brand: {
          50: '#E7F4FC',
          100: '#DDF2F8',
          500: '#12B8C2',
          600: '#0FA2AB',
          700: '#0C838A',
        },
        isl: {
          teal: '#14B8A6',
          cyan: '#06B6D4',
        },
        speech: {
          amber: '#F59E0B',
          orange: '#F97316',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-fast': 'pulse 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'ripple': 'ripple 1.5s linear infinite',
      }
    },
  },
  plugins: [],
};
export default config;
