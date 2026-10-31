const fs = require('fs');
const path = require('path');

function save(relPath, content) {
  const full = path.resolve(__dirname, '..', relPath);
  fs.mkdirSync(path.dirname(full), { recursive: true });
  fs.writeFileSync(full, content, 'utf8');
  console.log('Saved:', relPath);
}

// 1. apps/web/package.json
save('apps/web/package.json', `{
  "name": "@signbridge2vision/web",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev -p 3000",
    "build": "next build",
    "start": "next start -p 3000",
    "lint": "next lint"
  },
  "dependencies": {
    "@signbridge/types": "*",
    "clsx": "^2.1.1",
    "lucide-react": "^0.475.0",
    "next": "^15.1.7",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "tailwind-merge": "^3.0.1"
  },
  "devDependencies": {
    "@types/node": "^22.13.4",
    "@types/react": "^19.0.8",
    "@types/react-dom": "^19.0.3",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.5.2",
    "tailwindcss": "^3.4.17",
    "typescript": "^5.7.3"
  }
}`);

// 2. apps/web/tsconfig.json
save('apps/web/tsconfig.json', `{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}`);

// 3. apps/web/next.config.mjs
save('apps/web/next.config.mjs', `/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['@signbridge/types']
};

export default nextConfig;
`);

// 4. apps/web/postcss.config.mjs
save('apps/web/postcss.config.mjs', `export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
`);

// 5. apps/web/tailwind.config.ts
save('apps/web/tailwind.config.ts', `import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#0B0F19',
        surface: '#111827',
        'surface-elevated': '#1F2937',
        'surface-highlight': '#374151',
        brand: {
          50: '#EEF2FF',
          100: '#E0E7FF',
          500: '#6366F1',
          600: '#4F46E5',
          700: '#4338CA',
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
`);

// 6. apps/web/src/app/globals.css
save('apps/web/src/app/globals.css', `@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --background: #0B0F19;
  --foreground: #F3F4F6;
}

body {
  color: var(--foreground);
  background: var(--background);
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  overflow-x: hidden;
}

/* Custom Accessibility High-Contrast Scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #111827;
}

::-webkit-scrollbar-thumb {
  background: #374151;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #4B5563;
}

/* Focus styling for keyboard navigation accessibility */
*:focus-visible {
  outline: 2px solid #6366F1;
  outline-offset: 2px;
}
`);
