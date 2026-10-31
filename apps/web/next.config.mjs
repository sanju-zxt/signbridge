import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['@signbridge/types'],

  // Suppress the lockfile workspace warning
  outputFileTracingRoot: path.join(__dirname, '../../'),

  // Webpack config for Web Workers and TF.js
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Required for TF.js in browser
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }

    // Handle Web Workers via asset/resource
    config.module.rules.push({
      test: /\.worker\.(ts|js)$/,
      use: { loader: 'worker-loader', options: { filename: 'static/[hash].worker.js' } },
    });

    return config;
  },

  // Security headers without restrictive COEP blocking
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'Cross-Origin-Opener-Policy', value: 'same-origin-allow-popups' },
        ],
      },
    ];
  },
};

export default nextConfig;
