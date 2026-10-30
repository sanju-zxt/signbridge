import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SignBridge2Vision | Real-Time Multimodal AI Communication Platform',
  description: 'Bridging Indian Sign Language (ISL) users and spoken-language communicators through real-time temporal computer vision, streaming speech AI, and conversational intelligence.'
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-foreground antialiased selection:bg-brand-500 selection:text-white min-h-screen">
        {children}
      </body>
    </html>
  );
}
